#!/usr/bin/env python
# -*- coding: utf-8 -*-

with open('simserver/simserver.py') as f:
    for s in f.readlines():
        s = s.strip()
        if s.startswith('class'):
            print s
        if s.startswith('def'):
            print '\t' + s
