from collections import defaultdict, OrderedDict
import csv
from decimal import getcontext
from decimal import Decimal
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline


getcontext().prec = 14  # For Decimal


def convert_chunk_to_numpy(chunk):
    ts = np.array([float(x[0]) for x in chunk])
    acc_x = np.array([float(x[1]) for x in chunk])
    acc_y = np.array([float(x[2]) for x in chunk])
    acc_z = np.array([float(x[3]) for x in chunk])
    gyr_x = np.array([float(x[4]) for x in chunk])
    gyr_y = np.array([float(x[5]) for x in chunk])
    gyr_z = np.array([float(x[6]) for x in chunk])
    return (ts, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)


def smooth(x, y):
    xnew = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    return xnew, power_smooth


def get_linear_acceleration(
        acc_x,
        acc_y,
        acc_z):
    a_cgs = []
    for idx in range(len(acc_x)):
        a_x_mg = acc_x[idx]
        a_y_mg = acc_y[idx]
        a_z_mg = acc_z[idx]
        a_mg = np.array([a_x_mg, a_y_mg, a_z_mg])
        a_cg = a_mg # Without transform
        a_cgs.append(a_cg)
    return np.array(a_cgs)


def get_angular_acceleration(
        gyr_x,
        gyr_y,
        gyr_z,
        ts):
    a_angs = []
    d_omega_mg_dt_x = np.gradient(gyr_x, ts)
    d_omega_mg_dt_y = np.gradient(gyr_y, ts)
    d_omega_mg_dt_z = np.gradient(gyr_z, ts)
    for idx in range(len(gyr_x)):
        d_omega_mg_dt = np.array([d_omega_mg_dt_x[idx], d_omega_mg_dt_y[idx], d_omega_mg_dt_z[idx]])
        a_angs.append(d_omega_mg_dt)
    return np.array(a_angs)


MASS_OF_HEAD_KG = 5.0  # From: https://www.quora.com/How-much-does-an-average-human-head-weigh
G = 9.80665
G_TO_METERS_PER_SEC_SQUARED = G
METERS_PER_SEC_SQUARED_TO_G = 1.0 / G

def degrees_per_sec_to_rads_per_sec(degrees):
    return degrees * (math.pi / 180)


def convert_to_si_units(acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z):
    (acc_x, acc_y, acc_z) = (acc_x * G_TO_METERS_PER_SEC_SQUARED, acc_y * G_TO_METERS_PER_SEC_SQUARED, acc_z * G_TO_METERS_PER_SEC_SQUARED)
    gyr_x = degrees_per_sec_to_rads_per_sec(gyr_x)
    gyr_y = degrees_per_sec_to_rads_per_sec(gyr_y)
    gyr_z = degrees_per_sec_to_rads_per_sec(gyr_z)
    return (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)


def convert_to_g(lin_acc):
    return lin_acc * METERS_PER_SEC_SQUARED_TO_G


def handle_chunk(chunk, filename, idx):
    filename_prefix = os.path.basename(filename).split('.csv')[0]
    (ts, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z) = convert_chunk_to_numpy(chunk)

    (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z) = convert_to_si_units(acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)

    ##############################################################################
    # Linear acceleration
    ##############################################################################
    lin_acc = get_linear_acceleration(acc_x,
                                      acc_y,
                                      acc_z)

    lin_acc = convert_to_g(lin_acc)
    plt.figure()
    points = (ts, lin_acc)
    p = points[0] - points[0][0] # make sure times are relative to start of impact
    points = (p, points[1])
    plt.plot(*smooth(points[0], points[1][:, 0]), label='x')
    plt.plot(points[0], points[1][:, 0], '.', fillstyle='none')
    plt.plot(*smooth(points[0], points[1][:, 1]), label='y')
    plt.plot(points[0], points[1][:, 1], '.', fillstyle='none')
    plt.plot(*smooth(points[0], points[1][:, 2]), label='z')
    plt.plot(points[0], points[1][:, 2], '.', fillstyle='none')
    plt.title(f'Impact {idx+1} Linear acceleration')
    plt.xlabel('time (s)')
    plt.ylabel('g')
    plt.legend()
    plt.savefig(f'out/{filename_prefix}_{(idx+1):03}_linear.png', dpi=300)

    ##############################################################################
    # Angular acceleration
    ##############################################################################
    ang_acc = get_angular_acceleration(gyr_x,
                                       gyr_y,
                                       gyr_z,
                                       ts)

    plt.figure()
    points = (ts, ang_acc)
    p = points[0] - points[0][0]
    points = (p, points[1])
    plt.plot(*smooth(points[0], points[1][:, 0]), label='x')
    plt.plot(points[0], points[1][:, 0], '.', fillstyle='none')
    plt.plot(*smooth(points[0], points[1][:, 1]), label='y')
    plt.plot(points[0], points[1][:, 1], '.', fillstyle='none')
    plt.plot(*smooth(points[0], points[1][:, 2]), label='z')
    plt.plot(points[0], points[1][:, 2], '.', fillstyle='none')
    plt.title(f'Impact {idx+1} Angular Acceleration')
    plt.xlabel('time (s)')
    plt.ylabel('rad/s')

    plt.legend()
    plt.savefig(f'out/{filename_prefix}_{(idx+1):03}_angular.png', dpi=300)


def process_file(filename):
    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        input_dict = defaultdict(list)
        next(csvreader) # Skip header
        for idx, row in enumerate(csvreader):
            input_dict[row[0]].append(row[2:])
        # Spread out times
        spread_out_dict = OrderedDict()
        for k, v in input_dict.items():
            k_f = Decimal(k)
            if len(v) > 1:
                mult = Decimal(0.01) / len(v)
                for idx in range(len(v)):
                    spread_out_dict[k_f + mult * idx] = v[idx]
            else:
                spread_out_dict[k_f] = v[0]
        df = pd.DataFrame.from_dict(spread_out_dict, orient="index")
        df.to_csv(f"{filename}_chunked.csv")

        # Chunk up points
        chunks = []
        current_chunk = []
        previous_time = Decimal(0.0)
        for k in sorted(spread_out_dict.keys()):
            v = spread_out_dict[k]
            k_f = Decimal(k)
            time_diff = k_f - previous_time
            previous_time = k_f
            # Greater than 100 milliseconds seen as significant enough maybe?
            if time_diff > 0.1:
                if len(current_chunk) > 0:
                    chunks.append(current_chunk)
                current_chunk = []
            current_chunk.append([k_f] + spread_out_dict[k])
        for idx, chunk in enumerate(chunks):
            handle_chunk(chunk, filename, idx)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('filename', help='csv file to process')
    args = parser.parse_args()
    process_file(args.filename)
