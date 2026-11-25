%% Behaviour cloning of the human robot driver in THÖR-MAGNI (the learned driver)
%
% Input : bc_meta.mat + bc_partNN.mat (made by learning/data/make_bc_dataset.py)
%         x = 4 steps (1.6 s) x 34 features in a robot frame pointing at the 4 s
%         hindsight subgoal: own velocity (2), subgoal (2), 6 nearest people
%         [x(6) y(6) vx(6) vy(6) present(6)]; y = driver's mean velocity over the
%         next 0.8 s (forward, sideways). Test = recording day 17 May.
%
% Models
%   KEEP        keep the current velocity
%   GOAL-PACE   head for the subgoal at the driver's median speed (0.27 m/s)
%   BC-blind    LSTM on own motion + subgoal only (no people)
%   BC-social   LSTM on own motion + subgoal + people
%   BC-nocopy   LSTM on subgoal + people only, no own velocity
%   BC-people   LSTM on people only (frame still points at the subgoal)  <- the learned driver
%               (the subgoal's distance is the driver's 4 s progress, which also
%               gives away its speed; without it, speed has to come from the people)
%               (own past velocity lets the network copy itself, which makes it
%               stay slow once slowed in closed loop: the "copycat" problem)
%
% Questions: does BC-social predict what the human did better than the baselines,
% and do people matter (BC-social vs BC-blind), especially when someone is within
% 2 m and when the driver slowed down to yield (speed < 0.1 m/s)?
%
% Output (results_driver/): metrics.csv, driver.png, weights_BC_social.mat,
% weights_BC_blind.mat (plain arrays for the ROS 2 node, same layout as the predictor).

QUICK  = false;
SEEDS  = [1 2 3];
EPOCHS = 60;
OUT    = "results_driver";
if QUICK, SEEDS = 1; EPOCHS = 2; end
if ~isfolder(OUT), mkdir(OUT); end
D = loadParts();
fprintf("train %d, val %d, test %d samples\n", size(D.train_Y,1), size(D.val_Y,1), size(D.test_Y,1));

nearT = logical(D.test_near(:));
Yt = double(D.test_Y);
yieldT = sqrt(sum(Yt.^2, 2)) < 0.1;                     % driver (almost) stopped

%% Baselines
keep = squeeze(D.test_X(:, end, 1:2));
g = squeeze(D.test_X(:, end, 3:4));
goalpace = 0.27 * g ./ max(sqrt(sum(g.^2, 2)), 1e-6);
P.KEEP = keep; P.GOAL_PACE = goalpace;

%% Learned drivers
variants = ["BC-blind", "BC-social", "BC-nocopy", "BC-people"];
for v = variants
    cols = 1:4;
    if v == "BC-social", cols = 1:34; end
    if v == "BC-nocopy", cols = 3:34; end
    if v == "BC-people", cols = 5:34; end
    Xtr = toCells(D.train_X(:, :, cols)); Xva = toCells(D.val_X(:, :, cols)); Xte = toCells(D.test_X(:, :, cols));
    preds = zeros([size(Yt) numel(SEEDS)]);
    for k = 1:numel(SEEDS)
        rng(SEEDS(k));
        fprintf("\n== %s, seed %d ==\n", v, SEEDS(k));
        net = trainnet(Xtr, double(D.train_Y), buildNet(numel(cols)), "mse", options(Xva, double(D.val_Y), EPOCHS));
        preds(:, :, k) = minibatchpredict(net, Xte);
        if k == 1, nets.(matlab.lang.makeValidName(v)) = net; end
    end
    P.(matlab.lang.makeValidName(v)) = mean(preds, 3);
end

%% Metrics
names = string(fieldnames(P))';
rows = {};
for n = names
    E = P.(n) - Yt;
    vecErr = sqrt(sum(E.^2, 2));
    spd = sqrt(sum(P.(n).^2, 2)); spdT = sqrt(sum(Yt.^2, 2));
    spdErr = abs(spd - spdT);
    yieldRecall = mean(spd(yieldT) < 0.15);
    rows(end+1, :) = {n, mean(vecErr), mean(vecErr(nearT)), mean(spdErr), mean(spdErr(nearT)), ...
                      mean(spdErr(yieldT)), yieldRecall}; %#ok<SAGROW>
end
T = cell2table(rows, "VariableNames", ["model", "vel_err_ms", "vel_err_near_ms", "speed_err_ms", ...
                                       "speed_err_near_ms", "speed_err_when_yielding_ms", "yield_recall"]);
disp(T); writetable(T, fullfile(OUT, "metrics.csv"));
s = T.speed_err_near_ms;
fprintf("\nBC-social vs BC-blind near people: speed error %.1f%% lower\n", 100 * (1 - s(names == "BC_social") / s(names == "BC_blind")));
fprintf("BC-social vs best baseline near people: speed error %.1f%% lower\n", ...
    100 * (1 - s(names == "BC_social") / min(s(names == "KEEP" | names == "GOAL_PACE"))));
fprintf("Test samples: %d (%d near people, %d yielding)\n", numel(nearT), sum(nearT), sum(yieldT));

%% Figure: predicted vs true speed near people
f = figure("Visible", "off", "Position", [100 100 1000 380]);
try, f.Theme = "light"; catch, set(f, "Color", "w"); end
tiledlayout(1, 2);
nexttile; bar(categorical(T.model, T.model), [T.speed_err_ms T.speed_err_near_ms T.speed_err_when_yielding_ms]);
ylabel("speed error (m/s)"); legend("all", "person within 2 m", "driver yielding", "Location", "northwest");
title("How well each model predicts the human driver's speed");
nexttile; hold on; grid on;
spdT = sqrt(sum(Yt(nearT, :).^2, 2));
scatter(spdT, sqrt(sum(P.BC_social(nearT, :).^2, 2)), 8, "filled", "MarkerFaceAlpha", 0.4);
scatter(spdT, sqrt(sum(P.GOAL_PACE(nearT, :).^2, 2)), 8, "filled", "MarkerFaceAlpha", 0.2);
plot([0 0.8], [0 0.8], "k--"); xlabel("human driver speed (m/s)"); ylabel("predicted speed (m/s)");
legend("BC-social", "goal pace", "Location", "northwest"); title("Person within 2 m (test day)");
exportgraphics(f, fullfile(OUT, "driver.png"), "Resolution", 150);

%% Export plain weights (same layout as the predictor's)
for v = variants
    W = plainWeights(nets.(matlab.lang.makeValidName(v))); %#ok<NASGU>
    save(fullfile(OUT, "weights_" + strrep(v, "-", "_") + ".mat"), "-struct", "W", "-v7");
end
fprintf("Done. Results in %s/\n", OUT);

%% ---------------------------------------------------------------- helpers
function D = loadParts()
    D = load("bc_meta.mat");
    parts = dir("bc_part*.mat");
    if isempty(parts), error("Put bc_meta.mat and bc_partNN.mat next to this script."); end
    for p = sort(string({parts.name}))
        S = load(p);
        for f = string(fieldnames(S))'
            if isfield(D, f), D.(f) = cat(1, D.(f), S.(f)); else, D.(f) = S.(f); end
        end
    end
end

function X = toCells(A)                                  % N x T x F -> N cells of T x F
    X = squeeze(num2cell(permute(A, [2 3 1]), [1 2]));
end

function layers = buildNet(nFeat)
    layers = [
        sequenceInputLayer(nFeat, "Normalization", "zscore")
        lstmLayer(48, "OutputMode", "last")
        dropoutLayer(0.2)
        fullyConnectedLayer(48)
        reluLayer
        fullyConnectedLayer(2)];
end

function o = options(Xva, Yva, epochs)
    o = trainingOptions("adam", "MaxEpochs", epochs, "MiniBatchSize", 128, ...
        "InitialLearnRate", 2e-3, "L2Regularization", 1e-3, ...
        "LearnRateSchedule", "piecewise", "LearnRateDropPeriod", 20, "LearnRateDropFactor", 0.5, ...
        "Shuffle", "every-epoch", "ValidationData", {Xva, Yva}, "ValidationFrequency", 40, ...
        "OutputNetwork", "best-validation", "Plots", "none", "Verbose", true, "VerboseFrequency", 40);
end

function W = plainWeights(net)
    L = net.Learnables;
    get = @(layer, p) double(extractdata(L.Value{L.Layer == layer & L.Parameter == p}));
    names = string({net.Layers.Name});
    lstm = names(arrayfun(@(l) isa(l, "nnet.cnn.layer.LSTMLayer"), net.Layers));
    fcs  = names(arrayfun(@(l) isa(l, "nnet.cnn.layer.FullyConnectedLayer"), net.Layers));
    W.in_mean = double(net.Layers(1).Mean(:));
    W.in_std  = double(net.Layers(1).StandardDeviation(:));
    W.lstm_W = get(lstm, "InputWeights");  W.lstm_R = get(lstm, "RecurrentWeights");
    W.lstm_b = get(lstm, "Bias");
    W.fc1_W = get(fcs(1), "Weights"); W.fc1_b = get(fcs(1), "Bias");
    W.fc2_W = get(fcs(2), "Weights"); W.fc2_b = get(fcs(2), "Bias");
    W.notes = "x: 4 x 34 (see make_bc_dataset.py), z-scored; LSTM gates i,f,g,o; out = fc2(relu(fc1(h))) = [forward, sideways] m/s";
end
