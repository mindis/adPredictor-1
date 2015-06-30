__author__ = 'michaelpearmain'

'''
All imports are kept within the __init__ to afford the flexibility to reorganise the internal package structure without
worrying about side effects from internal submodule imports or the order imports within each module.

This helps make this package simple-to-grok for other programmers, and serve as a manifest of all functionality provided
by the package.

Currently the order of imports do not matter.

The use of numpy has tried to be avoided so this package can be easily used with pypy for fast execution.
'''
from exceptions import FSQError, FSQEnvError, FSQEncodeError,\
                       FSQTimeFmtError, FSQMalformedEntryError,\
                       FSQCoerceError, FSQEnqueueError, FSQConfigError,\
                       FSQPathError, FSQInstallError, FSQCannotLockError,\
                       FSQWorkItemError, FSQTTLExpiredError,\
                       FSQMaxTriesError, FSQScanError, FSQDownError,\
                       FSQDoneError, FSQFailError, FSQTriggerPullError,\
                       FSQHostsError, FSQReenqueueError, FSQPushError

import gzip
import random
import argparse
import json
import pickle
import logging
import mmh3

from collections import defaultdict
from sys import stderr
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt, erfc, pi