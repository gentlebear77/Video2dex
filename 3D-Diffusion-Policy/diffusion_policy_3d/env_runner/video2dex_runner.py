import wandb
import numpy as np
import torch
import tqdm
from diffusion_policy_3d.env.video2dex.ycb_relocate_env import YCBRelocate
from diffusion_policy_3d.env.video2dex.mug_place_object_env import MugPlaceObjectEnv
from diffusion_policy_3d.env.video2dex.mug_pour_water_env import WaterPouringEnv

from diffusion_policy_3d.gym_util.mjpc_diffusion_wrapper import MujocoPointcloudWrapperAdroit
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint

class Spec:
    def __init__(self, env=None, env_name="relocate-mug-1"):
        self.observation_dim = env.reset().shape[0]
        self.action_dim = env.action_spec[0].shape[0]
        self.env_id = env_name


class Video2dexRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_point_crop=True,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        env_info = task_name.split('-')
        task = env_info[0]
        obj_name = env_info[1]
        steps_per_render = max(10 // fps, 1)

        def env_fn():
            if task == 'relocate':
                e = YCBRelocate(has_renderer=False, object_name=obj_name)
                spec = Spec(e, task_name)
            elif task == 'pour':
                e = WaterPouringEnv(has_renderer=False, tank_size=(0.15, 0.15, 0.12))
                spec = Spec(e, task_name)
            elif task == 'place':
                e = MugPlaceObjectEnv(has_renderer=False, mug_scale=1.7)
                spec = Spec(e, task_name)
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                   e),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn()

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_returns_train = []
        all_success_rates = []
        


        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Video {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
                
            # start rollout
            reward_sum = 0
            obs = env.reset()
            policy.reset()
            done = False
            num_goal_achieved = 0
            actual_step_count = 0
            while not done:
                # create obs dict
                np_obs_dict = {
                'obs': obs.astype(np.float32)
                }
                # print(np_obs_dict)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))
                # print(obs_dict)
                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                    
                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)
                # step env
                obs, reward, done, info = env.step(action)
                # all_goal_achieved.append(info['goal_achieved']
                reward_sum += reward
                done = np.all(done)
                actual_step_count += 1

            all_returns_train.append(reward_sum)
            all_success_rates = all_returns_train
            # all_success_rates.append(env.is_success())


        # log
        log_data = dict()
        
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        log_data[f'sim_video_eval'] = videos_wandb

        # clear out video buffer
        _ = env.reset()
        # clear memory
        videos = None
        del env

        return log_data
