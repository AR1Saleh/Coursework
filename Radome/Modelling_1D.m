clc; clear; close all;

% Constants
f               = linspace(76e9, 81e9, 3000);                                   % Frequency range 
c               = 3e8;                                                      % Speed of light 
lambda          = c ./ f;                                                   % Wavelength 
freeSph         = (2 * pi * f) / c;                                         % Free space Phase Factor

epsilon_r       = 2;                                                        % Relative permittivity of radome
tan_delta       = 4e-4;                                                     % Loss tangent for radome material
eta_rad         = (1 + 1i * (tan_delta / 2)) * 120 * pi / sqrt(epsilon_r);  % Wave impedance

% Attenuation and phase factor
attenuation     = (pi .* tan_delta ./ c) * sqrt(epsilon_r) .* f;    
phasefactor     = (2 .* pi ./ c) .* (1 + (tan_delta^2) / 8) * sqrt(epsilon_r) .* f;                         
wavenumber      = (attenuation + 1i*phasefactor);

% Radome dimensions
d               = 21 *(mean(lambda) / 2);                               % Radome Radius
thickness       = 15 * (mean(lambda) / (2 * sqrt(epsilon_r)));          % Thickness
reg_width       = 15e-2;                                                % Region Width

% Reflection and Transmission Coefficients
e1              = exp((wavenumber - 1i*freeSph)*thickness);
e2              = exp(-(wavenumber + 1i*freeSph)*thickness);
sum             = (120 * pi + eta_rad)^2;
prod            = 4 * 120 * pi * eta_rad;
diff            = (120 * pi - eta_rad)^2;

Reflection      = ((120*pi)^2 - (eta_rad)^2)*(e2 - e1) ./ ((sum)*e1 - diff*e2);
Transmission    = prod ./ ((sum .* e1) - (diff .* e2));

% Plot both Reflection and Transmission
figure;
hold on;
plot(f / 1e9, abs(Reflection), 'r', 'LineWidth', 1.5, 'DisplayName', 'Reflection $\Gamma^2$');
plot(f / 1e9, abs(Transmission), 'b', 'LineWidth', 1.5, 'DisplayName', 'Transmission $\tau^2$');
plot(f / 1e9, (1 - exp(-attenuation * thickness)), 'k', 'LineWidth', 1.5, 'DisplayName', 'Attenuation $e^{-\alpha d}$');
legend('Interpreter', 'latex'); 
hold off;

% Add title, labels, legend, and grid
title('Reflection, Transmission & Attenuation Coefficients');
xlabel('Frequency (GHz)');
grid on;
legend('show');

% Region
r               = linspace(0, reg_width, 3000);
E_fields        = zeros(length(f), length(r)); % Electric fields
H_fields        = zeros(length(f), length(r)); % Magnetic fields

for idx_f = 1:length(f)  
    k = wavenumber(idx_f);  % Wavenumber for the current frequency
    E0 = 5;                 % Incident electric field amplitude (V/m)
    for idx_r = 1:length(r)

        if r(idx_r) < d
            % Region before the radome
            E_fields(idx_f, idx_r) = E0 * (exp(-1i .* freeSph(idx_f) .* r(idx_r)) + Reflection(idx_f) .* exp(1i .* freeSph(idx_f) .* r(idx_r)));
            H_fields(idx_f, idx_r) = (E0 / 120 * pi) * (exp(-1i .* freeSph(idx_f) .* r(idx_r)) - Reflection(idx_f) .* exp(1i .* freeSph(idx_f) .* r(idx_r)));
        elseif r(idx_r) <= d + thickness
            % Region inside the radome
            E2f = Transmission(idx_f) * E0 * (sqrt(sum) / 240*pi) * exp((k - j*(freeSph(idx_f)))*(d+thickness));
            E2b = Transmission(idx_f) * E0 * (sqrt(diff) / 240*pi) * exp(-(k + j*(freeSph(idx_f)))*(d+thickness));
            E_fields(idx_f, idx_r) = E2f*exp(-k*r(idx_r)) + E2b*exp(k*r(idx_r));
            H_fields(idx_f, idx_r) = (E2f*exp(-k*r(idx_r)) - E2b*exp(-k*r(idx_r)))/eta_rad;
        else
            % Region after the radome
            E_fields(idx_f, idx_r) = E0  * Transmission(idx_f) * exp(-1i * freeSph(idx_f) * r(idx_r));
            H_fields(idx_f, idx_r) = (E0 / 120 * pi) * Transmission(idx_f) * exp(-1i * freeSph(idx_f) * r(idx_r));
        end
    
    end
end

% 3D Plot for Electric Field (|E|)
[R, F] = meshgrid(r * 1e2, f * 1e-9); 
figure;
surf(R, F, abs(E_fields) , 'EdgeColor', 'none');
title('Electric Field Intensity (|E|) Across the Radome');
xlabel('Radius (cm)');
ylabel('Frequency (GHz)');
zlabel('|E| (V/m)');
colorbar;
grid on;
view(3);

% 3D Plot for Magnetic Field (|H|)
figure;
surf(R, F, abs(H_fields), 'EdgeColor', 'none');
title('Magnetic Field Intensity (|H|) Across the Radome');
xlabel('Radius (cm)');
ylabel('Frequency (GHz)');
zlabel('|H| (A/m)');
colorbar;
grid on;
view(3);


