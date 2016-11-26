#!/bin/bash
mkdir -p $1/$1_rst
mv $1/dump* $1/$1_rst/
cp $1/spec/bin.dat $1/$1_rst
cp $1/spec/list.cat $1/$1_rst
cp $1/spec/node.dat $1/$1_rst
