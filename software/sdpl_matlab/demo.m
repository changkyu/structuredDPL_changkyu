clear all;
close all;

env_setting;

%% Experiments for Speed of Convergence 
% with Randomly generated synthetic data
%
ex01_synthetic;
show_result_1_eIter;
show_result_1_ETA;
%ex02_synthetic_missing;

%% Experiments for Accuracy
% with cube data
ex03_sfm_cube;
show_result_3_sfm_cube;
%ex05_sfm_cube_nodenoise;