import sys
import os
import dill
import json
import argparse
import torch
import random
import numpy as np
import pandas as pd
import csv

import time

np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=2, sci_mode=False)

sys.path.append("../../trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str)
parser.add_argument("--likely", help="full evaluate", default=False, action='store_true')
parser.add_argument("--full", help="full evaluate", default=False, action='store_true')
parser.add_argument('--contrastive_weight', help='', type=float, default=0.0)
parser.add_argument('--seed', help='manual seed to use, default is 123', type=int, default=123)
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model(model_dir, env, ts=100, weight=0.0, seed=None):
    model_registrar = ModelRegistrar(model_dir, 'cpu')

    prefix = 'w-{:.4f}-s-{:d}'.format(weight, seed)
    model_registrar.load_models(ts, prefix)

    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint, weight=args.contrastive_weight, seed=args.seed)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']

    with torch.no_grad():

        if args.likely:

            tic = time.time()

            ############### MOST LIKELY ###############
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_col_joint_batch_errors = np.empty((0, 56))
            eval_col_truth_batch_errors = np.empty((0, 56))
            eval_col_cross_batch_errors = np.empty((0, 56))
            print("-- Evaluating GMM Grid Sampled (Most Likely)")
            for i, scene in enumerate(scenes):
                print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")

                for t in tqdm(range(0, scene.timesteps, 10)):
                    timesteps = np.arange(t, t + 10)

                    predictions = eval_stg.predict(scene,
                                                   timesteps,
                                                   ph,
                                                   num_samples=1,
                                                   min_history_timesteps=7,
                                                   min_future_timesteps=12,
                                                   z_mode=False,
                                                   gmm_mode=True,
                                                   full_dist=True)  # This will trigger grid sampling

                    if not predictions:
                        continue

                    batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                           scene.dt,
                                                                           max_hl=max_hl,
                                                                           ph=ph,
                                                                           node_type_enum=env.NodeType,
                                                                           map=None,
                                                                           prune_ph_to_future=True,
                                                                           kde=False,
                                                                           col=True)

                    eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                    eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
                    try:
                        eval_col_joint_batch_errors = np.concatenate((eval_col_joint_batch_errors, np.stack(batch_error_dict[args.node_type]['col_joint'])))
                        eval_col_truth_batch_errors = np.concatenate((eval_col_truth_batch_errors, np.stack(batch_error_dict[args.node_type]['col_truth'])))
                        eval_col_cross_batch_errors = np.concatenate((eval_col_cross_batch_errors, np.stack(batch_error_dict[args.node_type]['col_cross'])))
                    except Exception as e:
                        print(batch_error_dict[args.node_type]['col_joint'])
                        print('evaluate', len(batch_error_dict[args.node_type]['col_joint']))

            toc = time.time()
            print('Top-1 Eval Elapsed: %s' % (toc - tic))

            # pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'ml'}
            #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_most_likely.csv'))
            # pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'ml'}
            #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_most_likely.csv'))
            # pd.DataFrame({'value': eval_col_joint_batch_errors, 'metric': 'col_joint', 'type': 'ml'}
                         # ).to_csv(os.path.join(args.output_path, args.output_tag + '_col_most_likely.csv'))

        else:

            ############### BEST OF 20 ###############

            tic = time.time()

            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_col_joint_batch_errors = np.empty((0, 56))
            eval_col_truth_batch_errors = np.empty((0, 56))
            eval_col_cross_batch_errors = np.empty((0, 56))
            eval_kde_nll = np.array([])
            print("-- Evaluating best of 20")
            for i, scene in enumerate(scenes):
                print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
                for t in tqdm(range(0, scene.timesteps, 10)):
                    timesteps = np.arange(t, t + 10)
                    predictions = eval_stg.predict(scene,
                                                   timesteps,
                                                   ph,
                                                   num_samples=20,
                                                   min_history_timesteps=7,
                                                   min_future_timesteps=12,
                                                   z_mode=False,
                                                   gmm_mode=False,
                                                   full_dist=False)

                    if not predictions:
                        continue

                    batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                           scene.dt,
                                                                           max_hl=max_hl,
                                                                           ph=ph,
                                                                           node_type_enum=env.NodeType,
                                                                           map=None,
                                                                           best_of=True,
                                                                           prune_ph_to_future=True,
                                                                           col=True)

                    eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                    eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
                    eval_col_joint_batch_errors = np.concatenate((eval_col_joint_batch_errors, np.stack(batch_error_dict[args.node_type]['col_joint'])))
                    eval_col_truth_batch_errors = np.concatenate((eval_col_truth_batch_errors, np.stack(batch_error_dict[args.node_type]['col_truth'])))
                    eval_col_cross_batch_errors = np.concatenate((eval_col_cross_batch_errors, np.stack(batch_error_dict[args.node_type]['col_cross'])))
                    eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))

            toc = time.time()
            print('Top-20 Eval Elapsed: %s' % (toc - tic))

            # pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'best_of'}
            #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_best_of.csv'))
            # pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'best_of'}
            #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_best_of.csv'))
            # pd.DataFrame({'value': eval_col_joint_batch_errors, 'metric': 'col_joint', 'type': 'best_of'}
            #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_col_best_of.csv'))
            # pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'best_of'}
            #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_kde_best_of.csv'))

        # collision density
        col_joint_interp = np.nanmean(eval_col_joint_batch_errors, axis=0)
        col_truth_interp = np.nanmean(eval_col_truth_batch_errors, axis=0)
        col_cross_interp = np.nanmean(eval_col_cross_batch_errors, axis=0)

        col_joint_step = col_joint_interp[:-1].reshape(-1,5).mean(axis=1)
        col_truth_step = col_truth_interp[:-1].reshape(-1,5).mean(axis=1)
        col_cross_step = col_cross_interp[:-1].reshape(-1,5).mean(axis=1)

        # collision cumulative
        col_joint_cumulative = [np.nanmean(eval_col_joint_batch_errors[:, :i*5+6].max(axis=1)) for i in range(11)]
        col_truth_cumulative = [np.nanmean(eval_col_truth_batch_errors[:, :i*5+6].max(axis=1)) for i in range(11)]
        col_cross_cumulative = [np.nanmean(eval_col_cross_batch_errors[:, :i*5+6].max(axis=1)) for i in range(11)]

        col_joint_cumulative = np.array(col_joint_cumulative)
        col_truth_cumulative = np.array(col_truth_cumulative)
        col_cross_cumulative = np.array(col_cross_cumulative)

        print(f'CKPT: {args.checkpoint}')
        print(f'ADE: {np.nanmean(eval_ade_batch_errors):.4f}')
        print(f'FDE: {np.nanmean(eval_fde_batch_errors):.4f}')
        print(f'COL: {col_joint_cumulative[2]*100:.4f}')
        # Note: the inference of the Trajectron++ is conditioned on the states of neighboring agents up to the observation time, but not on any steps that have already been predicted about the future. This design makes the model unaware of the latest states of the nearby agents and causes high collision rate at long horizon. As such, our evaluation of collision rate is focused on the first four prediction steps where the models still have access to relatively up-to-date information of the surrounding neighbors.

        logname = os.path.join(args.model, 'result_{}_{:.4f}.csv'.format(os.path.basename(args.model), args.contrastive_weight))
        with open(logname, mode='a') as logfile:
            log_writer = csv.writer(logfile, delimiter=',')
            log_writer.writerow([args.checkpoint, args.seed, np.nanmean(eval_ade_batch_errors), np.nanmean(eval_fde_batch_errors), col_joint_cumulative[2]])

        ############### MODE Z ###############
        if args.full:
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_kde_nll = np.array([])
            print("-- Evaluating Mode Z")
            for i, scene in enumerate(scenes):
                print(f"---- Evaluating Scene {i+1}/{len(scenes)}")
                for t in tqdm(range(0, scene.timesteps, 10)):
                    timesteps = np.arange(t, t + 10)
                    predictions = eval_stg.predict(scene,
                                                   timesteps,
                                                   ph,
                                                   num_samples=2000,
                                                   min_history_timesteps=7,
                                                   min_future_timesteps=12,
                                                   z_mode=True,
                                                   full_dist=False)

                    if not predictions:
                        continue

                    batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                           scene.dt,
                                                                           max_hl=max_hl,
                                                                           ph=ph,
                                                                           node_type_enum=env.NodeType,
                                                                           map=None,
                                                                           prune_ph_to_future=True)
                    eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                    eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
                    eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))

            pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'z_mode'}
                         ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_z_mode.csv'))
            pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'z_mode'}
                         ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_z_mode.csv'))
            pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'z_mode'}
                         ).to_csv(os.path.join(args.output_path, args.output_tag + '_kde_z_mode.csv'))


            ############### FULL ###############
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_kde_nll = np.array([])
            print("-- Evaluating Full")
            for i, scene in enumerate(scenes):
                print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
                for t in tqdm(range(0, scene.timesteps, 10)):
                    timesteps = np.arange(t, t + 10)
                    predictions = eval_stg.predict(scene,
                                                   timesteps,
                                                   ph,
                                                   num_samples=2000,
                                                   min_history_timesteps=7,
                                                   min_future_timesteps=12,
                                                   z_mode=False,
                                                   gmm_mode=False,
                                                   full_dist=False)

                    if not predictions:
                        continue

                    batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                           scene.dt,
                                                                           max_hl=max_hl,
                                                                           ph=ph,
                                                                           node_type_enum=env.NodeType,
                                                                           map=None,
                                                                           prune_ph_to_future=True)

                    eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                    eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
                    eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))

            pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'full'}
                         ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_full.csv'))
            pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'full'}
                         ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_full.csv'))
            pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'full'}
                         ).to_csv(os.path.join(args.output_path, args.output_tag + '_kde_full.csv'))
