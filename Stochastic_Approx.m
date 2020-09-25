function Stochastic_Approx = Stochastic_Approx(dydt, yt_actual_vec, y_initial, k_initial, step_size, step_count, optimize_k, chart_title, x_label, y_label)
    % dydt: @(y, k)
    % This variable should be a function that returns the derivative given y and k.

    % yt_actual_vec: @(t)
    % This variable should be a function that returns the actual (analytical) y value given t.
    % Must accept t as a vector or a scalar.
    % 
    % If this variable is not a function handle, then Stochastic_Approx() will not plot anything
    % for yt_actual_vec. If only the stochastic models are desired, set yt_actual_vec to a non-function value (i.e 0).
    % If yt_actual_vec is not a function but optimize_k is true, Stochastic_Approx() throws an error,
    % because k cannot be optimized without yt_actual_vec.

    % y_initial: double
    % This variable is the initial y value for the desired solution.
    % This is the value from which the Euler's method and RK2 processes will start.

    % k_initial: double
    % This variable is the initial k value from which MATLAB will optimize.
    % If optimize_k is false, then MATLAB will use k_initial in the Euler method and RK2 steps.

    % step_size: double
    % This variable is the Euler's method and RK2 step size.

    % step_count: integer
    % This variable is the number of steps taken in the Euler's method and RK2 plots.

    % optimize_k: boolean
    % If this variable is true, MATLAB will use k_initial as a starting point for finding the optimal k value
    % which minimizes the SSE between each stochastic approximation (either Euler's method or RK2) and the actual
    % values (determined by yt_actual_vec).
    % 
    % If this variable is false, MATLAB will simply use k_initial as the k value to use in the stochastic approximations.
    % 
    % This variable is optional (defaults to true).

    % chart_title: string
    % This variable will be displayed above the chart

    if ~exist('optimize_k', 'var')
        optimize_k = true;
    end

    if ~exist('chart_title', 'var')
        chart_title = "Model";
    end

    if ~exist('y_label', 'var')
        y_label = "y";
    end

        if ~exist('x_label', 'var')
        x_label = "t";
    end

    %% PRINT STUFF %%
    dydt
    yt_actual_vec
    y_initial
    k_initial
    step_size
    step_count
    optimize_k

    %% INIT STUFF %%
    % Stochastic functions for performing Euler's Method and RK2
    stochastic_euler = @(y_initial_, k_, step_size_, step_count_) stochastic(...
            @(previous_y) euler_step(dydt, previous_y, k_, step_size_), ...
            y_initial_, ...
            step_size_, ...
            step_count_);

    stochastic_euler_improved = @(y_initial_, k_, step_size_, step_count_) stochastic(...
        @(previous_y) rk2_step(dydt, previous_y, k_, step_size_), ...
        y_initial_, ...
        step_size_, ...
        step_count_);

    % Functions for computing SSE for Euler's Method and RK2
    if isa(yt_actual_vec,'function_handle')
        euler_sse_func = @(k_, y_initial_) sum_squared_error(...
            stochastic_euler(y_initial_, k_, step_size, step_count), ...
            yt_actual(yt_actual_vec, step_size, step_count));

        euler_improved_sse_func = @(k_, y_initial_) sum_squared_error(...
            stochastic_euler_improved(y_initial_, k_, step_size, step_count), ...
            yt_actual(yt_actual_vec, step_size, step_count));
    end

    %% END INIT %%
    if optimize_k
        if ~isa(yt_actual_vec,'function_handle')
            disp("Error: optimize_k is true, but yt_actual_vec is not a function handle! Cannot optimize without yt_actual_vec function!")
            return
        end

        % Finding optimial k value which minimizes SSE for both Euler's Method and RK2
        euler_k = fminsearch(@(k_) euler_sse_func(k_, y_initial), k_initial)
        euler_improved_k = fminsearch(@(k_) euler_improved_sse_func(k_, y_initial), k_initial)
    else
        euler_k = k_initial
        euler_improved_k = k_initial
    end

    % Using the optimized k values to perform Euler's Method and RK2
    y_vals_euler = stochastic_euler(y_initial, euler_k, step_size, step_count)
    y_vals_euler_improved = stochastic_euler_improved(y_initial, euler_improved_k, step_size, step_count)

    % Initializing time vector for the plot's x-axis
    t_vec = get_t_vec(step_size, step_count);

    if isa(yt_actual_vec,'function_handle')
        % Printing SSE for the optimized k values
        euler_sse = euler_sse_func(euler_k, y_initial)
        euler_improved_sse = euler_improved_sse_func(euler_improved_k, y_initial)

        % Computing the actual y values using the differential equation's analytical solution
        yt_actual_step_size = min([0.01, step_size])
        t_vals_actual = get_t_vec(yt_actual_step_size, (step_size * step_count) / (yt_actual_step_size));
        y_vals_actual = yt_actual_vec(t_vals_actual);
    end

    % Plot
    if isa(yt_actual_vec,'function_handle')
        euler_legend_label = "Euler's method (k = " + num2str(euler_k, "%.5f") + ", SSE = " + num2str(euler_sse, "%.10f") + ")";
        euler_improved_legend_label = "Euler's method improved (k = " + num2str(euler_improved_k, "%.5f") + ", SSE = " + num2str(euler_improved_sse, "%.10f") + ")";
    else
        euler_legend_label = "Euler's method (k = " + num2str(euler_k, "%.5f") + ")";
        euler_improved_legend_label = "Euler's method improved (k = " + num2str(euler_improved_k, "%.5f") + ")";
    end

    figure(1);
    if isa(yt_actual_vec,'function_handle')
        plot(t_vec, y_vals_euler, '--x', t_vec, y_vals_euler_improved, '--o', t_vals_actual, y_vals_actual, '-');
        legend(euler_legend_label, euler_improved_legend_label, "Actual");
    else
        plot(t_vec, y_vals_euler, '--x', t_vec, y_vals_euler_improved, '--o');
        legend(euler_legend_label, euler_improved_legend_label);
    end
    xlabel(x_label);
    ylabel(y_label);
    title(chart_title)
end

function out = euler_step(dydt, previous_y, k, step_size)
    slope = dydt(previous_y, k);
    out = previous_y + (slope * step_size);
end

function out = rk2_step(dydt, previous_y, k, step_size)
    slope_1 = dydt(previous_y, k);
    y_euler_step = previous_y + (slope_1 * step_size);

    slope_2 = dydt(y_euler_step, k);

    slope_final = (slope_1 + slope_2) / 2;

    out = previous_y + (slope_final * step_size);
end

function out = yt_actual(yt_actual_vec, step_size, count)
    out = yt_actual_vec(get_t_vec(step_size, count));
end

function out = stochastic(y, y_initial, step_size, count)
    t_vals = get_t_vec(step_size, count);
    y_vals = zeros(1, count);
    y_vals(1) = y_initial;

    for t = 2:length(t_vals)
        y_vals(t) = y(y_vals(t - 1));
    end

    out = y_vals;
end

function out = get_t_vec(step_size, count)
    out = 0 : step_size : step_size * (count);
end

function out = sum_squared_error(vec1, vec2)
    out = sum((vec1 - vec2) .^ 2);
end