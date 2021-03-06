# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import json
import torch_xla
import re


def _snake_case_to_camel_case(name):
  return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def get_counter_stats(filter=None):
  stats = dict()
  for counter_name in torch_xla.debug.metrics.counter_names():
    if filter and filter not in counter_name:
      continue
    stats[counter_name] = torch_xla.debug.metrics.counter_value(counter_name)
  return stats


def get_metrics(filter=None):
  stats = dict()
  for metric_name in torch_xla.debug.metrics.metric_names():
    if filter and filter not in metric_name:
      continue
    stats[metric_name] = torch_xla.debug.metrics.metric_data(metric_name)
  return stats


class Stats:

  def __init__(self, **entries):
    self.__dict__.update(entries)


def get_stats(as_object=False):
  stats_dict = get_counter_stats()
  if as_object:
    new_dict = dict()
    for key, value in stats_dict.items():
      new_dict[_snake_case_to_camel_case(key)] = value
    return Stats(**new_dict)
  else:
    return stats_dict


def print_stats(
    pretty=True,
    prefix_text=None,
    suffix_text=None,
    include_metrics=False,
    filter=None,
):
  print("=================")
  print("= Statistics")
  print("=================")
  if prefix_text:
    print(prefix_text)

  stats = get_counter_stats(filter=filter)
  if include_metrics:
    stats.update(get_metrics(filter=filter))

  if pretty:
    print(json.dumps(stats, indent=4, sort_keys=True))
  else:
    print(stats)

  if suffix_text:
    print(suffix_text)
  print("=================")
