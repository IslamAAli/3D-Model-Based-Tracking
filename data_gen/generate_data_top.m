%% Initialization
clc; clear; close all;

%% Camera matrix and initial points
focal_length    = 100; 
img_size        = [480, 640]; % rows, cols ==> height, width
principle_point = img_size/2;

% list of initial model points assunming position at the origin
pts_3d_init = [ -15, -15, -15;
                15, -15, -15; 
                15, 15, -15;
                -15, 15, -15;
                
                -15, -15, 120; 
                15, -15, 120;
                15, 15, 120;
                -15, 15, 120];
            
% visualize points
figure;
grid on; grid minor;

drawWireFrame(pts_3d_init', 'b');

% scatter3(pts_3d_init(:,1), pts_3d_init(:,2),pts_3d_init(:,3), 'filled'); hold on;
title('Initial points');


% initial pose 
t = [50;40;100];
r_ang = [0,0,0]*(pi/180);
r_cos = cos(r_ang);
r_sin = sin(r_ang);

rz = [r_cos(3) -r_sin(3) 0; r_sin(3) r_cos(3) 0 ; 0 0 1];
ry = [r_cos(2) 0 r_sin(2); 0 1 0 ; -r_sin(2) 0 r_cos(2)];
rx = [1 0 0 ; 0 r_cos(1) -r_sin(1); 0 r_sin(1) r_cos(1)];
r = rz * ry * rx ;

% transform points to initial pose 
t_form = [r t; 0 0 0 1];
pts_3d_hom = [pts_3d_init ones(size(pts_3d_init,1), 1)]';
pts_transform = t_form * pts_3d_hom;

drawWireFrame(pts_transform, 'r');

% make camera matrix to get 2D projections
K = [focal_length, 0, principle_point(1), 0;
     0, focal_length, principle_point(2), 0;
     0, 0, 1, 0];
 
 pts_2d_hom = K*pts_transform;
 pts_2d_hom = pts_2d_hom ./ [pts_2d_hom(3,:); pts_2d_hom(3,:); pts_2d_hom(3,:)];
 
 figure;
 for i=1:1:50
     t = t+[0;5;0];
     t_form = [r t; 0 0 0 1];
     pts_2d_hom = K*t_form * pts_3d_hom;
     pts_2d_hom = pts_2d_hom ./ [pts_2d_hom(3,:); pts_2d_hom(3,:); pts_2d_hom(3,:)];
     drawimage(pts_2d_hom, 'b', img_size);
     
     pause(0.2);
 end

