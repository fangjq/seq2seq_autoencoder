# -*- coding: utf-8 -*- #
# Author: Jiaquan Fang

import argparse
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input-dir', help='input directory')
    arg('--output', help='output file')
    args = parser.parse_args()

    filenames = [join(args.input_dir, f) for f in listdir(args.input_dir)
                 if isfile(join(args.input_dir, f))]

    with open(args.output, "w") as out_file:
        for filename in filenames:
            if filename.startswith("data/Baidu_Query"):
                with open(filename, "r") as in_file:
                    for line in in_file:
                        line = line.strip("\n")
                        out_file.write(line[:line.find("QueryResult:")])
                        out_file.write("\n")
