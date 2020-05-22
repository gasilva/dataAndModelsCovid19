#!/bin/bash
nohup python $1  > foo.out 2> foo.err < /dev/null &


