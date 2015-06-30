__author__ = 'michaelpearmain'

import utils
import gzip
import random
import argparse
import json
import pickle
import logging

from collections import defaultdict
from sys import stderr
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt, erfc, pi
##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class base_learner(object):

    def __init__(self):
        None

    @classmethod
    def from_file(cls, f):
        format = f.readline()[:-1]
        if format == "json_w_group_v1":
            return ftrl_proximal_group.from_file(f, format)
        elif format == "json_mu_sigma2_adpredictor_v1":
            return adpredictor.from_file(f, format)
        else:
            stderr.write("Unknown format %s.\n" % format)
            return None

    @classmethod
    def from_params(cls, args):
        # choose a learner from args

        stderr.write("Creating from:\n")
        for v in sorted(args):
            stderr.write("%s\t=> %s\n" % (v, str(args[v])))

        D = 2**args["bits"]
        args["D"] = D

        learner = adpredictor.from_params(args)

        return learner


class hash_learner(base_learner):
    # base class for learners

    def __init__(self, D, hash = utils.feature_hash):

        base_learner.__init__(self)
        self.D = D
        self.hash = feature_hash

    @classmethod
    def bias_indice():
        return 0

    def _indices(self, x):

        D = self.D
        hash = self.hash

        l = [hash("%s=%s" % (i,x[i]), D) for i in x]
        l.append(0)
        return l


    def _explain_ind(self, x):

        D = self.D
        hash = self.hash

        with_hash = lambda x, D: (x, hash(x, D))

        l = [with_hash("%s=%s" % (i,x[i]), D) for i in x]
        l.append(("$", 0))

        if self.interactions:
            k = x.keys()
            v = x.values()
            L = len(k)
            for i in xrange(0, L):
                l.extend([with_hash("%s=%s+%s=%s" % (k[i], v[i], k[j], v[j]), D)
                          for j in xrange(i+1, L)])

        return l


    def explain(self, x, explanation, i = 0, not_ignored = None):

        for (f,h) in self._explain_ind(x):
            explanation.update(i, f, h, not_ignored)

class adpredictor(hash_learner):

    def __init__(self, mu, sigma2, D, args):

        hash_learner.__init__(self, D)

        self.mu = mu
        self.sigma2 = sigma2
        self.beta = args["ad_beta"]
        self.beta2 = self.beta ** 2
        self.epsilon = args["ad_epsilon"]
        self.args = args

        self.sigma0 = args.get("ad_sigma0", 1.0)
        self.ad_alpha = args.get("ad_alpha", 1.0)
        self.ad_gamma = args.get("ad_gamma", 0.0)

        self.max_w_update = args.get("max_w_update", 100.0)
        self.slowdown_sigma = args.get("slowdown_sigma", 1.0)

    @classmethod
    def from_params(cls, args):
        D = args["D"]
#       beta= args["adbeta"]
#        prior_probability = args.get(["prior_probability"], 0.5)

        mus = [0.] * D  # prior weight means

        sigma0 = args.get("ad_sigma0", 1.0)
        sigma2_0 = sigma0 * sigma0
        sigma2s = [sigma2_0] * D # prior weigth variances

#        bias = cls.prior_bias()
#        (mus[bias], sigma2s[bias]) = \
#            prior_bias_weight(prior_probability, beta,
#                             0)        # replace num_features which I do not know where to get

        return cls(mus, sigma2s, D, args)


    @classmethod
    def from_file(cls, f, format = None):
        if format == None:
            format = f.readline()[:-1]
        if format == "json_mu_sigma2_adpredictor_v1":
            (args, mu, sigma2) = json.load(f)
            args = ascii_encode_dict(args)
            return cls(mu, sigma2, args["D"], args)
        else:
            stderr.write("Unrecognized format %s\n" % format)
            return None


    def write_to_file(self, f, format = "json_mu_sigma2_adpredictor_v1"):
        if format == "json_mu_sigma2_adpredictor_v1":
            stderr.write("Writing in format %s to file %s..." % (format, str(f)))
            f.write("%s\n" % format)
            json.dump((self.args, self.mu, self.sigma2),
                      f)
            stderr.write("...done\n")
        else:
            stderr.write("Unrecognized format %s\n" % format)

    def weight_list(self):
        return [self.mu]

    def _active_mean_variance(self, features, predicting_with_alpha = False):

        means = (self.mu[f] for f in features)

        if predicting_with_alpha:
            if self.ad_alpha == 0.:
                return sum(means), self.beta2
            else:
                variances = (self.sigma2[f] for f in features)
                return sum(means), sum(variances) * self.ad_alpha + self.beta2
        else:
            variances = (self.sigma2[f] for f in features)
            return sum(means), sum(variances) + self.beta2

    def _active_mean_variance_with_gamma(self, features, predicting_with_alpha = False):

        gamma = self.ad_gamma

        means = (self.mu[f] / (1 + gamma * sqrt(self.sigma2[f])) for f in features)

        if predicting_with_alpha:
            if self.ad_alpha == 0.:
                return sum(means), self.beta2
            else:
                variances = (self.sigma2[f] for f in features)
                return sum(means), sum(variances) * self.ad_alpha + self.beta2
        else:
            variances = (self.sigma2[f] for f in features)
            return sum(means), sum(variances) + self.beta2


    def predict(self, x, ind = None):

        ind = ind if ind != None else self._indices(x)

        if self.ad_gamma == 0:
            total_mean, total_variance = \
                self._active_mean_variance(ind, predicting_with_alpha = True)
        else:
            total_mean, total_variance = \
                self._active_mean_variance_with_gamma(ind, predicting_with_alpha = True)

        # first formula is from adpredictor article, second is from adpredictor.py!!
        y_hat = norm_cdf(total_mean / sqrt(total_variance))
#        y_hat = norm_cdf(total_mean / total_variance)

        return y_hat


    def _apply_dynamics(self, w_mu, w_sigma2):

        if self.epsilon == 0.:
            return w_mu, w_sigma2

        else:

            prior_mu, prior_sigma2 = (0.0, self.sigma0 * self.sigma0) # prior_weight()

            adjusted_variance = w_sigma2 * prior_sigma2 / \
                ((1.0 - self.epsilon) * prior_sigma2 +
                 self.epsilon * w_sigma2)
            adjusted_mean = adjusted_variance * (
                (1.0 - self.epsilon) * w_mu / w_sigma2 +
                self.epsilon * prior_mu / prior_sigma2)

            return adjusted_mean, adjusted_variance


    def update(self, x, y, ind = None):

        ind = ind if ind != None else self._indices(x)

        y = 2. * y - 1. # to get y in [-1, 1]

        max_w_update = self.max_w_update
        slowdown_sigma = self.slowdown_sigma

        # magic adPredictor update
        total_mean, total_variance = self._active_mean_variance(ind)
        v, w = gaussian_corrections (y * total_mean / sqrt(total_variance))

        for f in ind:
            mu = self.mu[f]
            sigma2 = self.sigma2[f]
            mean_delta = y * sigma2 / sqrt(total_variance) * v
            variance_multiplier = 1.0 - (sigma2 / total_variance * w) * slowdown_sigma
            new_mu = mu + max(-max_w_update, min(max_w_update, mean_delta))
            new_sigma2 = sigma2 * variance_multiplier

            (self.mu[f], self.sigma2[f]) = \
                self._apply_dynamics(new_mu, new_sigma2)


class explanation(object):

    def __init__(self, n):
        self.ws = [defaultdict(int) for i in xrange(n)]
        self.nbs = [defaultdict(int) for i in xrange(n)]

    def update(self, l, f, h, not_ignored):

        if not_ignored != None:
            ws = not_ignored[l]
            if ws[h] == 0:
                return

        self.ws[l][f] = h
        self.nbs[l][f] += 1

    def write_explanation(self, fn):
        f = gzip.open(fn, 'wb') if fn[-3:] == ".gz" else open(fn,'wb')
        json.dump((self.ws, self.nbs), f)
        f.close()


def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


def data(f_train, D,
         dayfilter = None, dayfeature = True,
         counters = False,
         hcounters = False,
         group_hours = 1.,
         limit_values = None,
         feature_maps_dir = "",
         args = {}):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    device_ip_counter = defaultdict(int)
    device_id_counter = defaultdict(int)
    currday = 0

    device_ip_hcounter = None
    device_id_hcounter = None
    currhour = ""

    to_hour_string = ["%02d" % (i / group_hours) for i in xrange(24)]

    # efficiency reasons: I'd love to do it on all, but a static list should be faster
    feature_maps_features = ["device_ip", "device_id"]
    feature_maps_day = "00"
    feature_map = None

    user_app_site = args.get("user_app_site", False)
    user_domain = args.get("user_domain", False)

    if user_app_site:
        stderr.write("Using user_app_site flag\n")
        user_app_site_ip_counter = defaultdict(int)
        user_app_site_id_counter = defaultdict(int)
        user_app_site_currday = 0

    if user_domain:
        stderr.write("Using user_domain flag\n")

    no_hour_day_feature = args.get("no_hour_day_feature", False)
    if no_hour_day_feature:
        stderr.write("Using no_hour_day_feature\n")

    if limit_values != None:
        lim_did = set(limit_values["device_id"])
        lim_dip = set(limit_values["device_ip"])


    for t, row in enumerate(DictReader(f_train)):
        # process id
        ID = row['id']
        del row['id']

        # process clicks
        y = 0.
        if 'click' in row:
            if row['click'] == '1':
                y = 1.
            del row['click']

        # turn hour really into hour, it was originally YYMMDDHH
        date = row['hour'][0:6]
#        if group_hours == 1.:
#            row['hour'] = row['hour'][6:]
#        else:
#            row['hour'] = "%02d" % (int(row['hour'][6:]) / group_hours)

        row["hour"] = to_hour_string [int(row["hour"][6:])]

        if dayfilter != None and not date in dayfilter:
            # skip sample if this date is filtered out
            continue

        if dayfeature:
            # extract date
            row['wd'] = str(int(date) % 7)

            if not no_hour_day_feature:
                row['wd_hour'] = "%s_%s" % (row['wd'], row['hour'])

        if counters == True:
            d_ip = row['device_ip']
            d_id = row["device_id"]
            device_ip_counter[d_ip] += 1
            device_id_counter[d_id] += 1
            row["ipc"] = str(min(device_ip_counter[d_ip], 8))
            row["idc"] = str(min(device_id_counter[d_id], 8))
        elif counters == "daily":
            if currday < int(date):
                stderr.write("..new days is %s, resetting counters\n" % date)
                currday = int(date)
                device_ip_counter = defaultdict(int)
                device_id_counter = defaultdict(int)
            d_ip = row['device_ip']
            d_id = row["device_id"]
            device_ip_counter [d_ip] += 1
            device_id_counter [d_id] += 1
            row["ipc"] = str(min(device_ip_counter[d_ip], 8))
            row["idc"] = str(min(device_id_counter[d_id], 8))

        if hcounters == True:
            if currhour != row["hour"]:
                currhour = row["hour"]
                device_ip_hcounter = defaultdict(int)
                device_id_hcounter = defaultdict(int)

            d_ip = row['device_ip']
            d_id = row["device_id"]
            device_ip_hcounter[d_id] += 1
            device_id_hcounter[d_ip] += 1
            row["ipch"] = str(min(device_ip_hcounter[d_ip],4))
            row["idch"] = str(min(device_id_hcounter[d_id],4))

        if user_app_site:

            if user_app_site_currday != date:
                user_app_site_ip_counter = defaultdict(int)
                user_app_site_id_counter = defaultdict(int)
                user_app_site_currday = date
                stderr.write("..new day for user_app_site is %s, resetting counters\n" % \
                             user_app_site_currday)

            d_ip = row['device_ip']
            d_id = row["device_id"]

            if row["site_id"] == "85f751fd": # null value, we work on app
                p_app = d_ip + "+a=" + row["app_id"]
                d_app = d_id + "+a=" + row["app_id"]

                user_app_site_ip_counter [p_app] += 1
                user_app_site_id_counter [d_app] += 1
                row["ipa"] = str(min(user_app_site_ip_counter[p_app], 8))
                row["ida"] = str(min(user_app_site_id_counter[d_app], 8))

            else:
                p_site = d_ip + "+s=" + row["site_id"]
                d_site = d_id + "+s=" + row["site_id"]

                user_app_site_ip_counter [p_site] += 1
                user_app_site_id_counter [d_site] += 1
                row["ips"] = str(min(user_app_site_ip_counter[p_site], 8))
                row["ids"] = str(min(user_app_site_id_counter[d_site], 8))

        if user_domain:
            d_ip = row['device_ip']
            d_id = row["device_id"]

            if row["site_id"] == "85f751fd": # null value
                p_app = d_ip + "d=" + row["app_domain"]
                d_app = d_id + "d=" + row["app_domain"]

                device_ip_counter [p_app] += 1
                device_id_counter [d_app] += 1
                row["ipad"] = str(min(device_ip_counter[p_app], 8))
                row["idad"] = str(min(device_id_counter[d_app], 8))

            else:
                p_site = d_ip + "d=" + row["site_domain"]
                d_site = d_id + "d=" + row["site_domain"]

                device_ip_counter [p_site] += 1
                device_id_counter [d_site] += 1
                row["ipsd"] = str(min(device_ip_counter[p_site], 8))
                row["idsd"] = str(min(device_id_counter[d_site], 8))


        if feature_maps_dir != "":
            if date != feature_maps_day:
                feature_maps_day = date
                feature_map = None
                # load the feature map for the correct date - the day before!
                feature_map = load_feature_map(feature_maps_dir + '/' + \
                                               to_feature_map_date(date) + ".mfm")

            for r in feature_maps_features:
                fm = feature_map[r].get(row[r], None)
                if fm != None:
                    # count class
                    row[r + "*c"] = str(fm[0])
                    # proba class
                    row[r + "*p"] = str(fm[1])

        if limit_values != None:
#            for v in limit_values:
#                if not v in row:
#                    continue
#                if not row[v] in limit_values[v]:
#                    row[v] = "o"
#            if not row["device_id"] in limit_values["device_id"]:
#                del row["device_id"]
#            if not row["device_ip"] in limit_values["device_ip"]:
#                del row["device_ip"]
            if not row["device_id"] in lim_did:
                row["device_id"] = "o"
            if not row["device_ip"] in lim_dip:
                row["device_ip"] = "o"

        yield t, ID, row, y


##############################################################################
# start training #############################################################
##############################################################################


def myargs():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description =
"""
Perform training and prediction based on FTRL Optimal algorithm, with dropout added.
\nUsage is via:
\n
\n\t* Training:
\n
\n\t\tpypy fastd.py train -t <train set> -o <output model> --<various parameters>
\n
\n\t* Predicting:
\n
\n\t\tpypy fastd.py predict --test <test set> -i <input model> -p <output predictions>
\n
""")
    parser.add_argument('action', type=str,
                        help='action to perform: train   / predict')
    parser.add_argument('-t', "--train", default = "/dev/stdin")
    parser.add_argument('--test', default = "/dev/stdin")
    parser.add_argument('-p', "--predictions", default = "/dev/stdout")
    parser.add_argument("-o", "--outmodel")
    parser.add_argument("-i", "--inmodel")
    parser.add_argument('--alpha', default = 0.01, type = float)
    parser.add_argument('--beta', default = 2, type = float)
    parser.add_argument('--L1', default = 0.1, type = float)
    parser.add_argument('--L2', default = 0.01, type = float)
    parser.add_argument('--dropout', default = 0.8, type = float)
    parser.add_argument('--bits', default = 23, type = int)
    parser.add_argument('--n_epochs', default = 1, type = int)
    parser.add_argument('--holdout', default = 100, type = int)
    parser.add_argument("--interactions", action = "store_true")
    parser.add_argument("--sparse", action = "store_true")
    parser.add_argument("-v", '--verbose', default = 3, type = int)
    parser.add_argument('--onlydays', default = None,  type = str,
                        help = "restrict to specific, comma-separated days")
    parser.add_argument("--nodayfeature", action = "store_true")
    parser.add_argument("--device_counters", action = "store_true",
                        help = "use device_ip and device_id counter as features")
    parser.add_argument("--daily_device_counters", action = "store_true",
                        help = "use device_ip and device_id counter as features, daily only")
    parser.add_argument("--true_dropped", action = "store_true")
    parser.add_argument("--model_format", default = "json_w_group_v1", type = str)
    parser.add_argument("--seed", default = 1234, type = int)
    parser.add_argument("--group_fn", default = "one", type = str)
    parser.add_argument("--explain", default = None, type = str)
    parser.add_argument("--delayed_learning_factor", default = 0, type = float)
    parser.add_argument("--learner_type", default = "ftrl_group", type = str,
                        help = "type of learner, can be: ftrl_group, adpredictor")
    parser.add_argument("--ad_beta", default = 0.05, type = float,
                        help = "beta parameter for adpredictor")
    parser.add_argument("--ad_epsilon", default = 0.05, type = float,
                        help = "epsilon parameter for adpredictor")
    parser.add_argument("--ad_sigma0", default = 1.0, type = float,
                        help = "initial std dev parameter for adpredictor")
    parser.add_argument("--ad_alpha", default = 1.0, type = float,
                        help = "prediction time parameter for adpredictor, determines the importance of sigma's in prediction")
    parser.add_argument("--group_hours", default = 1.0, type = int,
                        help = "group hours: 1 = hour per hour, 2 = per 2 hours, and so on")
    parser.add_argument("--hourly_counters", action = "store_true")
    parser.add_argument("--ad_gamma", default = 0.0, type = float)
    parser.add_argument("--limit_values", default = "", type = str)
    parser.add_argument("--min_learning_rate", default = 1e-50, type = float)
    parser.add_argument("--feature_maps_dir", default = "", type = str,
                        help = "directory containing one xx.fm.json.gz feature map file per xx day")
    parser.add_argument("--max_w_update", default = 100., type = float)
    parser.add_argument("--slowdown_sigma", default = 1.0, type = float)
    parser.add_argument("--user_app_site", action = "store_true")
    parser.add_argument("--user_domain", action = "store_true")
    parser.add_argument("--no_hour_day_feature", action = "store_true")

    args = parser.parse_args()

    return args


def load_limits(limit_file):
    stderr.write("Loading limit file %s..." % limit_file)
    with open(limit_file) as f:
        d = ascii_encode_dict(json.load(f))
    for v in d:
        d[v] = set(d[v])
    stderr.write("... loaded\n")
    return d


def to_feature_map_date(currdate):
    # quick hack to get the day before the current date
    return currdate[0:4] + str(int(currdate[4:6]) - 1)


def load_feature_map(lfm):
    stderr.write("Loading feature map %s..." % lfm)
    with gzip.open(lfm) as f:
        d = ascii_encode_dict(json.load(f))
    stderr.write("keys: %s" % str(d.keys()))
    stderr.write("... loaded\n")
    return d


def train_learner(train, dayfilter, args, learner = None):

    if args.verbose > 1:
        stderr.write("Learning from %s\n" % train)

    start = datetime.now()

    holdout = args.holdout

    if learner == None:
        learner = base_learner.from_params(args.learner_type, vars(args))
    myargs = learner.args

    # we get some parameters from the learner
    dayfeaturep = learner.args["dayfeature"]
    counters = learner.args["device_counters"]
    D = learner.args["D"]
    group_hours = learner.args.get("group_hours",1.)
    hcounters = learner.args.get("hourly_counters", False)
    feature_maps_dir = learner.args.get("feature_maps_dir", "")

    limit_values = None if args.limit_values == "" else load_limits(args.limit_values)

        # start training
    for e in xrange(args.n_epochs):
        loss = 0.
        count = 0
        next_report = 10000
        c = 0

        if train[-3:] == ".gz":
           f_train = gzip.open(train, "rb")
        else:
           f_train = open(train)

        for t, ID, x, y in data(f_train, D,
                               dayfilter = dayfilter,
                               dayfeature = dayfeaturep,
                               counters = counters,
                               hcounters = hcounters,
                               group_hours = group_hours,
                               limit_values = limit_values,
                               feature_maps_dir = feature_maps_dir,
                               args = myargs):
           # data is a generator
           #  t: just a instance counter
           # ID: id provided in original data
           #  x: features
           #  y: label (click)

           if holdout != 0 and t % holdout == 0:
                # step 2-1, calculate holdout validation loss
                #           we do not train with the holdout data so that our
                #           validation loss is an accurate estimation of
                #           the out-of-sample error
               p = learner.predict(x)
               loss += logloss(p, y)
               count += 1
           else:
               # step 2-2, update learner with label (click) information
               learner.update(x, y)

           c += 1
           if args.verbose > 2 and c >= next_report:
               if count > 0:
                   stderr.write(' %s\tencountered: %d/%d\tcurrent logloss: %f in %s\n' % (
                        datetime.now(), c, t, loss/count, str(datetime.now()-start)))
               else:
                   stderr.write(' %s\tencountered: %d/%d\t no logloss calculation in %s\n' % (
                        datetime.now(), c, t, str(datetime.now()-start)))
               next_report *= 2

        if count != 0:
           stderr.write('Epoch %d finished, %d/%d samples per pass, holdout logloss: %f, elapsed time: %s\n' % (
                    e, c, t, loss/count, str(datetime.now() - start)))
        else:
           stderr.write('Epoch %d finished, %d/%d samples per pass, suspicious holdout logloss: %f/%f, elapsed time: %s\n' % (
                  e, c, t, loss, count, str(datetime.now() - start)))

        f_train.close()

    return learner


def predict_learner(learner, test, predictions, dayfilter, args):

    if args.verbose > 1:
        stderr.write("Predicting to %s with model %s ...\n" % (predictions, str(learner)))

    if test[-3:] == ".gz":
        f_test = gzip.open(test, "rb")
    else:
        f_test = open(test, "r")

    myargs = learner.args

    D = learner.args["D"]
    t = 0
    next_report = 1024.
    dayfeature = learner.args["dayfeature"]
    device_counters = learner.args["device_counters"]
    group_hours = learner.args.get("group_hours",1.)
    hcounters = learner.args.get("hourly_counters", False)
    feature_maps_dir = learner.args.get("feature_maps_dir", "")

    flv = learner.args.get("limit_values", "")
    limit_values = None if (flv == "" or flv == None) else load_limits(flv)

    # carefull, danger here!
    # the ad_alpha parameter from the command line overrides value from learner!
    stderr.write("Setting ad_alpha of learner to %.4f\n" % args.ad_alpha)
    learner.ad_alpha = args.ad_alpha
    stderr.write("Setting ad_gamma of learner to %.4f\n" % args.ad_gamma)
    learner.ad_gamma = args.ad_gamma

    if predictions[-3:] == ".gz":
        o = gzip.open
    else:
        o = open

    with o(predictions, 'wb') as outfile:
        outfile.write('id,click\n')
        for t, ID, x, y in data(f_test, D,
                                dayfilter = dayfilter,
                                dayfeature = dayfeature,
                                counters = device_counters,
                                hcounters = hcounters,
                                group_hours = group_hours,
                                limit_values = limit_values,
                                feature_maps_dir = feature_maps_dir,
                                args = myargs):

            p = learner.predict(x)
            outfile.write('%s,%.4f\n' % (ID, p))

            if args.verbose > 2:
                t += 1
                if t > next_report:
                    next_report *= sqrt(2)
                    stderr.write("...%d..." % t)

    f_test.close()

    if args.verbose > 1:
        stderr.write("Predicted.\n")


def explain_learner(learner, test, dayfilter, args):

    if args.verbose > 1:
        stderr.write("Explaining with model %s ...\n" % (str(learner)))

    if test[-3:] == ".gz":
        f_test = gzip.open(test, "rb")
    else:
        f_test = open(test, "r")

    myargs = learner.args

    D = learner.args["D"]
    n = learner.args.get("n",1)
    expl = explanation(n)
    not_ignored = learner.weight_list()
    feature_maps_dir = learner.args.get("feature_maps_dir", "")

    flv = learner.args.get("limit_values", "")
    limit_values = None if (flv == "" or flv == None) else load_limits(flv)

    t = 0
    next_report = 1024.

    for t, ID, x, y in data(f_test, D,
                            dayfilter = dayfilter,
                            dayfeature = learner.args["dayfeature"],
                            counters = learner.args["device_counters"],
                            hcounters = learner.args.get("hourly_counters", False),
                            group_hours = learner.args.get("group_hours",1.),
                            limit_values = limit_values,
                            feature_maps_dir = feature_maps_dir,
                            args = myargs):

        learner.explain(x, expl, not_ignored = not_ignored)

        if args.verbose > 2:
            t += 1
            if t > next_report:
                next_report *= sqrt(2)
                stderr.write("...%d..." % t)

    if args.verbose > 1:
        stderr.write("Explained.\n")

    return expl


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def main_fast_and_ugly(adict = None):

    args = myargs() if adict == None else Bunch(adict)

    learner = None

    if args.onlydays == None:
        dayfilter = None
    else:
        dayfilter = set(args.onlydays.split(","))
        stderr.write("Considering only days %s...\n" % str(dayfilter))

    if args.inmodel != None:
        with gzip.open(args.inmodel, "rb") as f:
            stderr.write("Loading model from %s\n" % args.inmodel)
            learner = base_learner.from_file(f)

    else:
        learner = None

    if args.action in ["train", "train_predict"]:
        random.seed(args.seed)
        learner = train_learner(args.train, dayfilter, args, learner)
        if args.outmodel != None and args.outmodel != "None":
            with gzip.open(args.outmodel, "wb") as f:
                learner.write_to_file(f)

    if args.action in ["predict", "train_predict"]:
        random.seed(args.seed)
        predict_learner(learner, args.test, args.predictions, dayfilter, args)

    if args.action in ["explain"]:
        expl = explain_learner(learner, args.test, dayfilter, args)
        learner = None                        # release memory!
        expl.write_explanation(args.explain)

    return learner


if __name__ == "__main__":
    main_fast_and_ugly()