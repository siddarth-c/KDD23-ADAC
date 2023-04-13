import pathlib
import os
import multiprocessing as mp


from resco_benchmark.multi_signal import MultiSignal
import argparse
from resco_benchmark.config.agent_config import agent_configs
from resco_benchmark.config.map_config import map_configs
from resco_benchmark.config.mdp_config import mdp_configs

from ADAC_traffic_master.ADAC.dac_mdp import dac_builder
from ADAC_traffic_master.ADAC.utils import ReplayBuffer

import numpy as np
import time


random_seeds = [i for i in range(3000, 4000, 5)]

# for settings in [['STOCHASTIC', 'NotADAC', 'Nil'], 
#                  ['MAXPRESSURE', 'NotADAC', 'Nil'], 
#                  ['STOCHASTIC', 'ADAC', 'Nil'], 
#                  ['STOCHASTIC', 'ADAC', 'Average_Cat']]:

for settings in [['STOCHASTIC', 'NotADAC', 'Nil'], 
                ['MAXPRESSURE', 'NotADAC', 'Nil']]:

# for settings in [['MAXPRESSURE', 'NotADAC', 'Nil']]:
    
    s1, s2, s3 = settings

    time_per_setting_S = time.time()

    print('<<' * 10, settings, '>>' * 10)


    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=str, default=s1,
                    choices=['STOCHASTIC', 'MAXWAVE', 'MAXPRESSURE', 'IDQN', 'IPPO', 'MPLight', 'MA2C', 'FMA2C',
                                'MPLightFULL', 'FMA2CFull', 'FMA2CVAL'])
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--eps", type=int, default=20)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--map", type=str, default='cologne8',
                    choices=['grid4x4', 'arterial4x4', 'ingolstadt1', 'ingolstadt7', 'ingolstadt21',
                                'cologne1', 'cologne3', 'cologne8',
                                ])
    ap.add_argument("--pwd", type=str, default = '/home///Name/RESCO_ADAC/RESCO_main/resco_benchmark')
    ap.add_argument("--log_dir", type=str, default=os.path.join('/home///Name/RESCO_ADAC/RESCO_main/resco_benchmark', 'results' + os.sep))
    ap.add_argument("--gui", type=bool, default=False)
    ap.add_argument("--libsumo", type=bool, default=False)
    ap.add_argument("--tr", type=int, default=0)  # Can't multi-thread with libsumo, provide a trial number

    ap.add_argument("--which", type=str, default=s2, choices=['ADAC', 'NotADAC'])
    ap.add_argument("--how", type=str, default=s3, choices=['Nil', 'Average', 'Average_Cat'])


    args = ap.parse_args()

    if args.libsumo and 'LIBSUMO_AS_TRACI' not in os.environ:
        raise EnvironmentError("Set LIBSUMO_AS_TRACI to nonempty value to enable libsumo")

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # MY EDITS START HERE

    ADAC_specific_S = time.time()

    if args.which == 'ADAC':


        def process_obs(obs):

            if args.how != 'Nil' and args.which == 'ADAC':

                what_type = 'list'
                processed = []

                if type(obs) is dict:

                    what_type = 'dict'
                    processed = {}
                    obs = np.array([obs[jn_name] for jn_name in jn_names])
                

                for i in range(len(jn_names)):

                    jn_name = jn_names[i]

                    if args.how == 'Average':

                        neighbours = [i] + [n - 1 for n in neighbour_list[i]]
                        averaged = np.mean(obs[neighbours], 0)

                    elif args.how == 'Average_Cat':

                        neighbours = [n - 1 for n in neighbour_list[i]]
                        averaged = np.mean(obs[neighbours], 0)
                        averaged = np.hstack((obs[i], averaged))
                    
                    if what_type == 'dict': 
                        processed[jn_name] = averaged
                    else:
                        processed.append(averaged)
                    
                if what_type == 'list':
                    processed = np.array(processed)
                
                return processed
            
            else:

                return obs
            

        buffer_dir_path = '/home///Name/RESCO_ADAC/RESCO_main/resco_benchmark/Buffer'

        if args.map == 'cologne3':

            jn_names = ['360082', '360086', 'GS_cluster_2415878664_254486231_359566_359576']
            neighbour_list = [[2], [1, 3], [2]]
        
        elif args.map == 'cologne8':

            jn_names = ['256201389', '280120513', '252017285', 'cluster_1098574052_1098574061_247379905', '26110729', '32319828', '62426694', '247379907']
            neighbour_list = [[2], [1, 5, 7], [4, 6, 7], [3, 8], [2, 8], [3], [2, 3], [4, 5]]

        elif args.map == 'ingolstadt7':
            jn_names = ['cluster_1757124350_1757124352', 'gneJ143', 'gneJ207', 'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190', '32564122', 'gneJ260', 'gneJ210']
            neighbour_list = [[2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6]]



        all_states = []
        all_next_states = []
        all_rewards = []
        all_actions =[ ]
        all_dones = []

        buffer_path = '/home/Name//Name/RESCO_ADAC/RESCO_main/resco_benchmark/Buffer/'



        for jn_name in jn_names:

            # print('Loading', jn_name, 'files.....')

            fname = buffer_path + jn_name + '_state.npy'
            c = np.load(fname)
            all_states.append(c)

            fname = buffer_path + jn_name + '_next_state.npy'
            c = np.load(fname)
            all_next_states.append(c)

            fname = buffer_path + jn_name + '_reward.npy'
            c = np.load(fname)
            c = [i[0] for i in c] # + np.random.randn()
            all_rewards.append(c)

            fname = buffer_path + jn_name + '_action.npy'
            c = np.load(fname)
            c = [i[0] for i in c]
            all_actions.append(c)

            fname = buffer_path + jn_name + '_not_done.npy'
            c = np.load(fname)
            c = [i[0] for i in c]
            all_dones.append(c)


        all_states = np.array(all_states)
        all_next_states = np.array(all_next_states)
        all_rewards = np.array(all_rewards)
        all_actions = np.array(all_actions)
        all_dones = np.array(all_dones)

        all_states = process_obs(all_states)
        all_next_states = process_obs(all_next_states)



        all_policies = {}

        ADAC_build_MDP_S = time.time()

        for states, actions, next_states, rewards, dones, jn_name in zip(all_states, all_actions, all_next_states, all_rewards, all_dones, jn_names):

            print('Building', jn_name, 'MDPs.....')

            num_actions = len(np.unique(actions))
            state_dim = states.shape[-1]

            buffer = ReplayBuffer(state_dim = state_dim, is_atari = False, atari_preprocessing = None, batch_size = 128, buffer_size = len(states), device = 'cpu')
            for (s, a, n, r, d) in zip(states, actions, next_states, rewards, dones):
                buffer.add(s, a, n, r, d, 0, 0)
            
            dac = dac_builder(num_actions, state_dim, buffer, None, 'cpu')
            policies = dac.get_policies()[0]

            all_policies[jn_name] = policies

        ADAC_build_MDP_E = time.time()


        def sample_action(obs):

            obs = process_obs(obs)

            action = {}

            for jn_name in jn_names:

                a = all_policies[jn_name].select_action(obs[jn_name])
                action[jn_name] = a
            
            return action
    
    ADAC_specific_E = time.time()

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # MY EDITS ENDS HERE



    def run_trial(args, trial):
        mdp_config = mdp_configs.get(args.agent)
        if mdp_config is not None:
            mdp_map_config = mdp_config.get(args.map)
            if mdp_map_config is not None:
                mdp_config = mdp_map_config
            mdp_configs[args.agent] = mdp_config

        agt_config = agent_configs[args.agent]
        agt_map_config = agt_config.get(args.map)
        if agt_map_config is not None:
            agt_config = agt_map_config
        alg = agt_config['agent']

        if mdp_config is not None:
            agt_config['mdp'] = mdp_config
            management = agt_config['mdp'].get('management')
            if management is not None:    # Save some time and precompute the reverse mapping
                supervisors = dict()
                for manager in management:
                    workers = management[manager]
                    for worker in workers:
                        supervisors[worker] = manager
                mdp_config['supervisors'] = supervisors

        map_config = map_configs[args.map]
        num_steps_eps = int((map_config['end_time'] - map_config['start_time']) / map_config['step_length'])
        route = map_config['route']
        if route is not None: route = os.path.join(args.pwd, route)
        if args.map == 'grid4x4' or args.map == 'arterial4x4':
            if not os.path.exists(route): raise EnvironmentError("You must decompress environment files defining traffic flow")

        if args.which == 'ADAC':
            path =  args.how + '_' + args.which +'-tr'+str(trial)
        else:
            path = alg.__name__+'-tr'+str(trial)

        env = MultiSignal(path,
                        args.map,
                        os.path.join(args.pwd, map_config['net']),
                        agt_config['state'],
                        agt_config['reward'],
                        route=route, step_length=map_config['step_length'], yellow_length=map_config['yellow_length'],
                        step_ratio=map_config['step_ratio'], end_time=map_config['end_time'],
                        max_distance=agt_config['max_distance'], lights=map_config['lights'], gui=args.gui,
                        log_dir=args.log_dir, libsumo=args.libsumo, warmup=map_config['warmup'])

        agt_config['episodes'] = int(args.eps * 0.8)    # schedulers decay over 80% of steps
        agt_config['steps'] = agt_config['episodes'] * num_steps_eps
        agt_config['log_dir'] = os.path.join(args.log_dir, env.connection_name)
        agt_config['num_lights'] = len(env.all_ts_ids)

        # Get agent id's, observation shapes, and action sizes from env
        obs_act = dict()
        for key in env.obs_shape:
            obs_act[key] = [env.obs_shape[key], len(env.phases[key]) if key in env.phases else None]
        agent = alg(agt_config, obs_act, args.map, trial)

        for e in range(args.eps):
            obs = env.reset(seed = random_seeds[e])
            done = False
            printed = False
            while not done:
                if args.which == 'ADAC':
                    act = sample_action(obs)

                    if e == 0 and printed is False:
                        print()
                        print(args.which, args.how)
                        print()
                        printed = True
                else:
                    act = agent.act(obs)
                obs, rew, done, info = env.step(act)
                agent.observe(obs, rew, done, info)
        env.close()

        return path


    if args.procs == 1 or args.libsumo:
        path = run_trial(args, args.tr)
    else:
        pool = mp.Pool(processes=args.procs)
        for trial in range(1, args.trials+1):
            pool.apply_async(run_trial, args=(args, trial))
        pool.close()
        pool.join()

    
    time_per_setting_E = time.time()

    with open (args.log_dir + '/stats/' + path + 'stats_rebuttal.txt', 'w') as file:
        file.write('Total test run hours:' + str(args.eps))
        file.write('\n')
        file.write('Total run time for this setting:' + str(time_per_setting_E - time_per_setting_S))  
        file.write('\n')
        file.write('ADAC specific time (loading + building MDPs):' + str(ADAC_specific_E - ADAC_specific_S))  
        file.write('\n')
        try:
            file.write('Time taken to build ADAC MDPs (only building MDPs):' + str(ADAC_build_MDP_E - ADAC_build_MDP_S))
        except:    
            file.write('\n')