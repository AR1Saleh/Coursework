clc; clear; close all;
% All the parameters and calculation results are shown on the command window.
%% Task 1: Motor Selection & Parameter Identification
m = 240;        % Total mass (kg)
v_target = 30;  % Target speed (km/h)
t_accel = 10;   % Acceleration time (s)
r_wheel = 0.25; % Wheel radius (m)
C_rr = 0.015;   % Rolling resistance coefficient
eta = 0.85;     % Drivetrain efficiency

% Convert speed to m/s
v = v_target * 1000 / 3600; % 8.33 m/s

% Acceleration force
a = v / t_accel;
F_accel = m * a;

% Rolling resistance force
F_rr = C_rr * m * 9.81;

% Total force and torque at wheel
F_total = F_accel + F_rr;
T_wheel = F_total * r_wheel;

% Gear ratio calculation
wheel_rpm = (v * 60) / (2 * pi * r_wheel);
gear_ratio = 4000 / wheel_rpm; % Motor RPM = 4000

% Motor torque requirements
T_motor_peak = T_wheel / gear_ratio;
P_peak = (F_total * v) / eta;

fprintf('Peak Motor Torque: %.2f Nm\n', T_motor_peak);
fprintf('Peak Power: %.2f kW\n', P_peak / 1000);

%% ================== BLDC MOTOR STATE-SPACE MODEL =======================
% BLDC Parameters (Golden Motor HPM3000B)
R = 0.1;        % Phase resistance (Ω)
L = 0.5e-3;     % Phase inductance (H)
Kt = 0.1176;    % Torque constant (Nm/A)
Ke = 0.1176;    % Back-EMF constant (V/(rad/s)) = Kt
J = 0.001;      % Rotor inertia (kg·m²)
b = 0.001;      % Damping coefficient (N·m·s/rad)

% BLDC State-Space Equations (2 states: [i; ω])
% dx/dt = A*x + B*u
%   x = [current (A); angular velocity (rad/s)]
%   u = [voltage (V); load torque (Nm)]
A = [-R/L  -Ke/L;
      Kt/J  -b/J];
B = [1/L   0;
      0   -1/J];
C = [0 1];  % Output = angular velocity (rad/s)
D = [0 0];

sys_bldc = ss(A, B, C, D);

% Transfer function from voltage to speed (ignoring load torque)
G_bldc = tf(sys_bldc(1,1));
G_bldc = minreal(G_bldc);

fprintf('BLDC Motor Transfer Function (Voltage to Speed):\n');
G_bldc

%% ================== OPEN-LOOP ANALYSIS ========================
T_load_max    = 5.22;
T_load_levels = T_load_max * [0.4; 0.6; 1];
t             = (0:0.01:10)';     % time vector

V_nominal = 48;
V_step    = V_nominal * (t>=1);         % step at t=1s
V_ramp    = V_nominal * (t/max(t));     % linear ramp

figure('Units','normalized','Position',[.1 .1 .8 .7]);

for k = 1:length(T_load_levels)
    T_load = T_load_levels(k);
    
    % Build inputs [voltage, load torque]
    U_step = [V_step,  T_load*ones(size(t))];
    U_ramp = [V_ramp,  T_load*ones(size(t))];
    
    % Simulate
    y_step = lsim(sys_bldc, U_step, t);
    y_ramp = lsim(sys_bldc, U_ramp, t);
    
    % Convert to RPM
    rpm_step = y_step * 60/(2*pi);
    rpm_ramp = y_ramp * 60/(2*pi);
    
    % --- STEP subplot ---
    ax1 = subplot(3,2,2*k-1);
    yyaxis left
    plot(t, rpm_step, 'b', 'LineWidth', 1.5);
    ylabel('Speed (RPM)')
    hold on

    yyaxis right
    plot(t, V_step, 'r--', 'LineWidth', 1);
    ylabel('Voltage (V)')
    
    title(sprintf('Step Input & Response (T_{load}=%.2f Nm)',T_load))
    xlabel('Time (s)')
    legend('Speed','Voltage','Location','best')
    grid on
    
    % --- RAMP subplot ---
    ax2 = subplot(3,2,2*k);
    yyaxis left
    plot(t, rpm_ramp, 'b', 'LineWidth', 1.5);
    ylabel('Speed (RPM)')
    hold on

    yyaxis right
    plot(t, V_ramp, 'r--', 'LineWidth', 1);
    ylabel('Voltage (V)')
    
    title(sprintf('Ramp Input & Response (T_{load}=%.2f Nm)',T_load))
    xlabel('Time (s)')
    legend('Speed','Voltage','Location','best')
    grid on
end

sgtitle('Open‑Loop Speed (RPM) & Voltage Inputs Under Varying Loads')

%% ======== Closed-Loop Controller Design with Speed Profile =========
% Design PI controller with performance constraints
opts = pidtuneOptions('PhaseMargin', 65, 'DesignFocus', 'reference-tracking');
C_pi = pidtune(G_bldc, 'PI', opts);

% Closed-loop system
sys_cl = feedback(C_pi * G_bldc, 1);

% Performance metrics
step_info = stepinfo(sys_cl);
fprintf('PI Controller Gains:\n Kp = %.5f, Ki = %.5f\n', C_pi.Kp, C_pi.Ki);
fprintf('Settling Time (2%%): %.2f s\nOvershoot: %.1f%%\n', ...
        step_info.SettlingTime, step_info.Overshoot);

% Speed Profile Generation (10 segments over 120 seconds)
t_total = 120; % Total time = 2 minutes
t = 0:0.01:t_total; % Time vector (10 ms steps)

% Define key times and speeds for the speed profile
key_times = [0, 10, 25, 35, 50, 60, 75, 85, 95, 110, 120]; % in seconds
key_speeds = [0, 1000, 1000, 2000, 2000, 1500, 1500, 0, -1000, -1000, 0]; % in RPM

% Generate reference speed using interpolation
ref_speed = interp1(key_times, key_speeds, t, 'linear');

% Convert RPM to rad/s (for simulation)
ref_speed_rads = ref_speed * 2 * pi / 60;

% Simulate closed-loop response
[y, t] = lsim(sys_cl, ref_speed_rads, t);

% Convert output back to RPM
y_rpm = y * 60 / (2 * pi);

% Plot results
figure;
plot(t, ref_speed, '--', t, y_rpm, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Speed (RPM)');
title('Speed Profile Tracking (PI Controller)');
legend('Reference', 'Actual');
grid on;

%% ================ Load Disturbance Rejection ==================

R = 0.1;        % Phase resistance (Ω)
L = 0.5e-3;     % Phase inductance (H)
Kt = 0.1176;    % Torque constant (Nm/A)
Ke = 0.1176;    % Back-EMF constant (V/(rad/s))
J = 0.001;      % Rotor inertia (kg·m²)
b = 0.001;      % Damping coefficient (N·m·s/rad)

% State-space matrices
A = [-R/L, -Ke/L;
     Kt/J, -b/J];
B = [1/L, 0;
     0, -1/J];
C = [0, 1];  % Output is angular velocity (ω)

% Extract motor input matrices
B1 = B(:,1);  % Input for voltage (V)
B2 = B(:,2);  % Input for load torque (T_load)

% PI Controller Parameters
Kp = C_pi.Kp;
Ki = C_pi.Ki;

% Augment state-space for closed-loop system
% States: [x; xi], Inputs: [ref_speed; T_load], Output: ω
A_cl = [A - B1*Kp*C, B1*Ki;
       -C,           0];
B_cl = [B1*Kp, B2;
        1,     0];
C_cl = [C, 0];
D_cl = [0, 0];

% Create closed-loop MIMO system
G_cl = ss(A_cl, B_cl, C_cl, D_cl);

% Simulation setup
t = 0:0.001:10;              % Time vector (0 to 10s)
ref_speed_rads = 418.9;      % Reference speed (4000 RPM = 418.9 rad/s)
T_load_max = 5.22;              % Max load torque 
T_load = 0.4*T_load_max * ones(size(t));  % 40% load initially
T_load(t >= 5) = 0.8*T_load_max;          % Step to 80% at t=5s

% Input matrix: [ref_speed; T_load]
u = [ref_speed_rads*ones(size(t))', T_load'];

% Simulate closed-loop response
[y_rads, ~] = lsim(G_cl, u, t);

% Convert to RPM for plotting
y_rpm = y_rads * 60 / (2*pi);
ref_speed_rpm = ref_speed_rads * 60 / (2*pi);

% Plot results
figure;
plot(t, ref_speed_rpm*ones(size(t)), '--r', t, y_rpm, 'b', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Speed (RPM)');
title('Load Disturbance Rejection (40% → 80% Load at t=5s)');
legend('Reference', 'Actual', 'Location', 'best');
grid on;

%% =============== PID Compensator Design for BLDC Motor =================

% Transfer function from voltage to speed (ignoring load torque)
G_bldc = tf(sys_bldc(1,1));

% Design PID Controller
opts = pidtuneOptions('PhaseMargin', 45, 'DesignFocus', 'reference-tracking');
Comp = pidtune(G_bldc, 'PID', opts);

% Analyze the compensated system
sys_open = Comp * G_bldc;
figure;
margin(sys_open); % Plot Bode diagram to verify phase margin

% Closed-Loop Analysis
sys_cl = feedback(Comp * G_bldc, 1);
% Performance metrics
step_info = stepinfo(sys_cl);
fprintf('PID Controller Gains:\n Kp = %.5f, Ki = %.5f\n, Kd = %.5f\n', C_pi.Kp, C_pi.Ki, C_pi.Kd);
fprintf('Settling Time (2%%): %.2f s\nOvershoot: %.1f%%\n', ...
        step_info.SettlingTime, step_info.Overshoot);

% Bode Plot Comparison
figure;
bode(G_bldc, sys_open, sys_cl);
legend('Uncompensated', 'PID Compensated (OL)', 'PID Compensated (CL)');
title('Bode Plot Comparison');
grid on;

[mag, phase, w] = bode(sys_cl); 
mag_db = 20*log10(squeeze(mag)); 
w_3db = interp1(mag_db, w, -3); % Find -3 dB bandwidth
disp(['Closed-loop bandwidth: ', num2str(w_3db), ' rad/s']);

% Nyquist Plot Comparison
figure; 
nyquist(G_bldc, sys_open, sys_cl);
legend('Uncompensated', 'PID Compensated (OL)', 'PID Compensated (CL)');
title('Nyquist Plot Comparison');
grid on;


%% ================== Buck Converter Design ======================

% ================== BUCK CONVERTER PARAMETERS ==================
Vin = 96;       % Input voltage (V)
L = 200e-6;     % Inductance (H)
C = 500e-6;     % Capacitance (F)
R_load = 10;    % Nominal load (Ω)
D = 0.5;        % Duty cycle (for 48V output)

% ================== STATE-SPACE MODEL ==================
A = [0       -1/L;
     1/C    -1/(R_load*C)];
B = [Vin/L; 0];
C = [0 1];
D_mat = 0;
sys_buck = ss(A, B, C, D_mat);

% Transfer function
[num, den] = ss2tf(A, B, C, D_mat);
G_buck = tf(num, den);
fprintf('Buck Converter Transfer Function:\n');
G_buck

% ================== SIMULATION CASES ==================
% Case 1: Step change in input voltage (96V → 72V at t=0.05s)
t = 0:1e-5:0.1; % Time vector
u1 = [0.5*ones(5000,1); 0.5*(72/96)*ones(5001,1)]; % Duty cycle adjusted

% Case 2: Load step change (10Ω → 5Ω at t=0.05s)
R_load_step = [10*ones(5000,1); 5*ones(5001,1)];
u2 = 0.5*ones(size(t))'; % Fixed duty cycle

% Simulate responses
[y1, t1] = lsim(sys_buck, u1, t);
[y2, t2] = lsim(sys_buck, u2, t, [], R_load_step);

% ================== PLOTTING ==================
figure;
subplot(2,1,1);
plot(t1, y1, 'b', 'LineWidth', 1.5);
title('Input Voltage Step (96V → 72V)');
xlabel('Time (s)'); ylabel('Output Voltage (V)'); grid on;

subplot(2,1,2);
plot(t2, y2, 'r', 'LineWidth', 1.5);
title('Load Step (10Ω → 5Ω)');
xlabel('Time (s)'); ylabel('Output Voltage (V)'); grid on;

%% ================== Integrated Control System ====================

% ================== SYSTEM COMPONENTS ==================
% Buck Converter Transfer Function (from Task 7)
s = tf('s');
G_buck = 96 / (1e-7*s^2 + 0.002*s + 1); 

% BLDC Motor Transfer Function (from Task 2)
G_bldc = 235200 / (s^2 + 201*s + 2.786e04);  

% Combined System (for PID tuning)
G_combined = series(G_buck, G_bldc);

% ================== PI CONTROLLER TUNING ==================
% Tune PI for combined system with 70° phase margin
opts = pidtuneOptions('PhaseMargin', 70, 'DesignFocus', 'reference-tracking');
C_pi = pidtune(G_combined, 'PI', opts);

% Closed-loop system (for step response analysis)
sys_cl_tf = feedback(C_pi * G_combined, 1);

% Performance metrics
step_info = stepinfo(sys_cl_tf);
fprintf('PI Controller Gains:\n Kp = %.5f, Ki = %.5f\n', C_pi.Kp, C_pi.Ki);
fprintf('Settling Time (2%%): %.2f s\nOvershoot: %.1f%%\n', ...
        step_info.SettlingTime, step_info.Overshoot);

% ================== SPEED PROFILE TEST ==================
t_total = 120; % Total time = 2 minutes
t = 0:0.01:t_total; % Time vector (10 ms steps)

% Define key times and speeds for the speed profile
key_times = [0, 10, 25, 35, 50, 60, 75, 85, 95, 110, 120]; % in seconds
key_speeds = [0, 1000, 1000, 2000, 2000, 1500, 1500, 0, -1000, -1000, 0]; % in RPM

% Generate reference speed using interpolation
ref_speed = interp1(key_times, key_speeds, t, 'linear');

% Convert RPM to rad/s (for simulation)
ref_speed_rads = ref_speed * 2 * pi / 60;

% Simulate closed-loop response
[y, t] = lsim(sys_cl_tf, ref_speed_rads, t);

% Convert output back to RPM
y_rpm = y * 60 / (2 * pi);

% Plot results
figure;
plot(t, ref_speed, '--', t, y_rpm, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Speed (RPM)');
title('Speed Profile Tracking (PI Controller)');
legend('Reference', 'Actual');
grid on;

% ======================== LOAD DISTURBANCE TEST =========================

% --- Parameters ---
% Buck Converter Parameters
Vin = 96;           % Input voltage (V)
L = 200e-6;         % Buck converter inductance (H)
C = 500e-6;         % Capacitance (F)

% BLDC Motor Parameters
R = 0.1;            % Phase resistance (Ω)
L_motor = 0.5e-3;   % Motor phase inductance (H)
Kt = 0.1176;        % Torque constant (Nm/A)
Ke = 0.1176;        % Back-EMF constant (V/(rad/s))
J = 0.001;          % Rotor inertia (kg·m²)
b = 0.001;          % Damping coefficient (N·m·s/rad)

% PI Controller Gains (example values, replace with your tuned values)
Kp = C_pi.Kp;       % Proportional gain
Ki = C_pi.Ki;       % Integral gain

% ====== Integrated Model of the Converter, Motor, and Controller ======

% --- State-Space Matrices ---
% State vector: x = [i_L (buck inductor current), v_C (buck capacitor voltage), 
%                    i (motor current), omega (motor speed), x_i (integrator state)]^T
% Inputs: u = [r (reference speed in rad/s), T_load (load torque in Nm)]^T
% Output: y = omega (motor speed in rad/s)

% A_cl matrix
A_cl = [
    0,          -1/L,       0,          -Vin/L*Kp,  Vin/L*Ki;   % di_L/dt
    1/C,        0,         -1/C,       0,          0;           % dv_C/dt
    0,          1/L_motor, -R/L_motor, -Ke/L_motor, 0;          % di/dt
    0,          0,         Kt/J,      -b/J,       0;            % domega/dt
    0,          0,         0,         -1,         0             % dx_i/dt
];

% B_cl matrix for inputs [r, T_load]
B_cl = [
    Vin/L*Kp,   0;          % r effect on i_L
    0,          0;          
    0,          0;          
    0,          -1/J;       % T_load effect on omega
    1,          0           % r effect on x_i
];

% Define output matrices (y = omega)
C_cl = [0, 0, 0, 1, 0];
D_cl = [0, 0];

% Create state-space system
sys_cl = ss(A_cl, B_cl, C_cl, D_cl);

% --- Simulation ---
% Time vector
t = 0:0.0001:10;  % 10 seconds with 0.1ms steps

% Reference speed (4000 RPM, converted to rad/s)
ref_speed_rpm = 4000 * ones(size(t));
ref_speed_rads = ref_speed_rpm * 2*pi / 60;

% Load torque (0 Nm initially, 0.5 Nm at t >= 5s)
T_load_max = 5.22;              % Max load torque 
T_load = 0.4*T_load_max * ones(size(t));  % 40% load initially
T_load(t >= 5) = 0.8*T_load_max;          % Step to 80% at t=5s

% Input matrix: [r; T_load] (transpose for lsim)
u = [ref_speed_rads; T_load]';

% Simulate response
[y, t_sim, x] = lsim(sys_cl, u, t);

% Convert output to RPM for plotting
y_rpm = y * 60 / (2*pi);

% Plot results
figure;
plot(t, y_rpm, 'b', t, ref_speed_rpm, 'r--', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Speed (RPM)');
title('Integrated System Response with PI Controller');
legend('Actual Speed', 'Reference Speed');
grid on;