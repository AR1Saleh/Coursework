clc; clear; close all;

% Constants
f = 76.5e9;                                                   % Frequency
c = 3e8;                                                      % Speed of light
lambda = c ./ f;                                              % Wavelength
freeSph = (2 * pi) / lambda;                                  % Free space Phase Factor

epsilon_r = 2;                                                % Relative permittivity of radome
tan_delta = 4e-4;                                             % Loss tangent for radome material
eta_rad = (1 + 1i * (tan_delta / 2)) * 120 * pi / sqrt(epsilon_r); % Wave impedance

% Attenuation and phase factor
attenuation = (pi .* tan_delta ./ c) * sqrt(epsilon_r) .* f;
phasefactor = (2 .* pi ./ c) .* (1 + (tan_delta^2) / 8) * sqrt(epsilon_r) .* f;
wavenumber = (attenuation + 1i*phasefactor);

% Radome dimensions
thickness = 15 * (mean(lambda) / (2 * sqrt(epsilon_r)));   % Thickness

% Reflection and Transmission Coefficients
e1              = exp((wavenumber - 1i*freeSph)*thickness);
e2              = exp(-(wavenumber + 1i*freeSph)*thickness);
sum             = (120 * pi + eta_rad)^2;
prod            = 4 * 120 * pi * eta_rad;
diff            = (120 * pi - eta_rad)^2;

Reflection      = ((120*pi)^2 - (eta_rad)^2).*(e2 - e1) ./ ((sum).*e1 - diff.*e2);
Transmission    = prod ./ ((sum .* e1) - (diff .* e2));

% Geometry of Radome
d = 21 * (lambda / 2);                              % Radome radius (R >> λ/2π & R = nλ/2)
r = linspace(0, 50e-2, 500);                        % Radial distances (m), 0.6mm step
theta = linspace(-pi / 2, pi / 2, 500);             % Elevation angles (rad)
phi = linspace(0, 2 * pi, 500);                     % Azimuth angles (rad)
[Theta, Phi] = meshgrid(theta, phi);

% Patch Antenna Field
eps_s = 2.2;
W  = (0.5 * lambda) / sqrt(eps_s);
L  = W;                                                                    
wx = (L .* sin(Theta) .* cos(Phi)) / lambda;
wy = (W .* sin(Theta) .* sin(Phi)) / lambda;
f  = abs(cos(pi * wx) .* sinc(pi * wy)); % Patch antenna
E0 = 5 * ((cos(Theta) .* sin(Phi)).^2 + cos(Phi).^2) .* f; % Incident E-field (V/m)

% Initialize field arrays
[R, Theta, Phi] = meshgrid(r, theta, phi); 
E_fields = zeros(size(R)); % Electric fields
H_fields = zeros(size(R)); % Magnetic fields

% Compute fields for all regions
for idx_r = 1:length(r)
    if r(idx_r) < d
        % Region before the radome
        E_fields(idx_r, :, :) = E0 .* (exp(-1i .* freeSph .* r(idx_r)) + Reflection .* exp(1i .* freeSph .* r(idx_r)));
        H_fields(idx_r, :, :) = (E0 ./ 120 * pi) .* (exp(-1i .* freeSph .* r(idx_r)) - Reflection .* exp(1i .* freeSph .* r(idx_r)));
    elseif r(idx_r) <= d + thickness
        % Region inside the radome
        E2f = Transmission * E0 * (sqrt(sum) / 240*pi) * exp((wavenumber - j*(freeSph))*(d+thickness));
        E2b = Transmission * E0 * (sqrt(diff) / 240*pi) * exp(-(wavenumber + j*(freeSph))*(d+thickness));
        E_fields(idx_r, :, :) = E2f*exp(-wavenumber*r(idx_r)) + E2b*exp(wavenumber*r(idx_r));
        H_fields(idx_r, :, :) = (E2f*exp(-wavenumber*r(idx_r)) - E2b*exp(-wavenumber*r(idx_r)))/eta_rad;
    else
        % Region after the radome
        E_fields(idx_r, :, :) = E0 .* Transmission .* exp(-1i * freeSph .* r(idx_r));
        H_fields(idx_r, :, :) = (E0 ./ 120 * pi) .* Transmission .* exp(-1i * freeSph .* r(idx_r));
    end
end

% Extract fields for a fixed radius
fixed_r_idx = round(length(r));
Er = squeeze(E_fields(fixed_r_idx, :, :)); % Electric field for fixed radius
Hr = squeeze(H_fields(fixed_r_idx, :, :)); % Magnetic field for fixed radius

% Radiation Patterns
figure;
subplot(1, 1, 1);
polarplot(theta, abs(E0(round(end/2), :)), 'b', 'LineWidth', 1.5);
hold on;
polarplot(theta, abs(Er(round(end/2), :)), 'r--', 'LineWidth', 1.5);
title('Radiation Pattern Comparison');
legend('Without Radome', 'With Radome');
grid on;

% 3D Radiation Pattern for Fixed r
figure;
subplot(1, 2, 1);
surf(theta , phi , abs(Er), 'EdgeColor', 'none');
title('Electric Field Intensity (|E|) for Fixed Radius');
xlabel('\theta (deg)');
ylabel('\phi (deg)');
zlabel('|E| (V/m)');
colorbar;
view(3);

subplot(1, 2, 2);
surf(theta , phi , abs(Hr), 'EdgeColor', 'none');
title('Magnetic Field Intensity (|H|) for Fixed Radius');
xlabel('\theta (deg)');
ylabel('\phi (deg)');
zlabel('|H| (A/m)');
colorbar;
view(3);
