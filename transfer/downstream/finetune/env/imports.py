
import sys
import os
import pdb
import time
import random
from uuid import uuid4

import json
import shutil
import tempfile
import codecs
from functools import wraps
from hashlib import sha256
from io import open
import boto3
import requests
from botocore.exceptions import ClientError
from tqdm import tqdm
import logging
import tarfile
from os.path import isfile, join
from os import listdir

from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence


try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

import pickle
import argparse

from collections import OrderedDict, Iterable

import re

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import git
from tensorboardX import SummaryWriter

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn import linear_model
from scipy.stats import hmean


