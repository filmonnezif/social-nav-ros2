%% Train and evaluate pedestrian trajectory predictors on THÖR-MAGNI (H1, H2)
%
% Input : thor_magni_sc3_snippets.mat (made by learning/data/make_dataset.py)
%         8 observed steps (3.2 s) -> 12 future steps (4.8 s) at 2.5 Hz, in the
%         person's own frame. Test set = recording day 17 May, never trained on.
%
% Models
%   CV            constant velocity (last observed velocity), no learning
%   LSTM-ego      own past motion only
%   LSTM-social   + 4 nearest people
%   LSTM-robot    + 4 nearest people + the teleoperated robot
%
% H1: LSTM beats CV by >= 15% ADE at 4.8 s on the test day.
% H2: with the robot within 3 m, LSTM-robot beats LSTM-social.
%
% Output (folder results_predictor/): metrics.csv, metrics_per_run.csv,
%   results.mat, ade_fde.png, examples.png, best model as .mat (+ .onnx when
%   the ONNX converter is installed).
%
% Runs on CPU in MATLAB Online; set QUICK = true for a short smoke test.

QUICK  = false;
SEEDS  = [1 2 3];          % each LSTM variant is trained once per seed
EPOCHS = 30;
DATA   = "thor_magni_sc3_snippets.mat";
OUT    = "results_predictor";

if QUICK, SEEDS = 1; EPOCHS = 1; end
if ~isfolder(OUT), mkdir(OUT); end
D = loadData(DATA);
fprintf("train %d, val %d, test %d snippets\n", size(D.train_fut,1), size(D.val_fut,1), size(D.test_fut,1));
if QUICK
    D = subsetSplit(D, "train", 2000); D = subsetSplit(D, "val", 500); D = subsetSplit(D, "test", 1000);
end

dt = 0.4; nObs = 8; nFut = 12;
variants = ["LSTM-ego", "LSTM-social", "LSTM-robot"];
testRuns = string(cellstr(D.test_runs));

%% Constant-velocity baseline
predCV = constantVelocity(D.test_ego, nFut, dt);
[adeCV, fdeCV] = errors(predCV, D.test_fut);

%% Train LSTM variants
preds = struct();
for v = variants
    Xtr = features(D, "train", v); Xva = features(D, "val", v); Xte = features(D, "test", v);
    Ytr = flatFuture(D.train_fut); Yva = flatFuture(D.val_fut);
    P = zeros([size(D.test_fut) numel(SEEDS)], "single");
    for k = 1:numel(SEEDS)
        rng(SEEDS(k));
        fprintf("\n== %s, seed %d ==\n", v, SEEDS(k));
        net = trainnet(Xtr, Ytr, buildNet(size(Xtr{1},2), nFut), "mse", ...
            options(Xva, Yva, EPOCHS));
        P(:,:,:,k) = unflatFuture(minibatchpredict(net, Xte), nFut);
        if k == 1, nets.(matlab.lang.makeValidName(v)) = net; end
    end
    preds.(matlab.lang.makeValidName(v)) = P;
end

%% Metrics: all test snippets and the near-robot subset, mean over seeds
near = logical(D.test_near(:));
rows = {};
rows(end+1,:) = {"CV", mean(adeCV), mean(fdeCV), mean(adeCV(near)), mean(fdeCV(near)), 0, 0};
perSeed = struct();
for v = variants
    P = preds.(matlab.lang.makeValidName(v));
    A = zeros(numel(near), numel(SEEDS)); F = A;
    for k = 1:numel(SEEDS), [A(:,k), F(:,k)] = errors(P(:,:,:,k), D.test_fut); end
    perSeed.(matlab.lang.makeValidName(v)) = A;
    rows(end+1,:) = {v, mean(A,"all"), mean(F,"all"), mean(A(near,:),"all"), mean(F(near,:),"all"), ...
                     std(mean(A,1)), std(mean(A(near,:),1))}; %#ok<SAGROW>
end
T = cell2table(rows, "VariableNames", ["model","ADE_m","FDE_m","ADE_near_robot_m","FDE_near_robot_m", ...
                                        "ADE_seed_std_m","ADE_near_seed_std_m"]);
disp(T); writetable(T, fullfile(OUT, "metrics.csv"));

%% Per test run (4 runs on 17 May): does the ranking hold in every run?
runIdx = runIndex(D, testRuns);
R = table();
for r = 1:numel(testRuns)
    m = runIdx == r; mn = m & near;
    row = table(testRuns(r), mean(adeCV(m)), 'VariableNames', ["run","ADE_CV"]);
    for v = variants
        A = mean(perSeed.(matlab.lang.makeValidName(v)), 2);
        row.("ADE_" + strrep(v,"-","_")) = mean(A(m));
        row.("ADE_near_" + strrep(v,"-","_")) = mean(A(mn));
    end
    R = [R; row]; %#ok<AGROW>
end
disp(R); writetable(R, fullfile(OUT, "metrics_per_run.csv"));

%% Hypotheses
bestLSTM = T.ADE_m(2:end); [~, b] = min(bestLSTM); bestName = variants(b);
h1 = 1 - T.ADE_m(1+b) / T.ADE_m(1);
aSoc = mean(perSeed.LSTM_social(near,:), 2); aRob = mean(perSeed.LSTM_robot(near,:), 2);
h2 = 1 - mean(aRob) / mean(aSoc);
h2runs = R.ADE_near_LSTM_robot < R.ADE_near_LSTM_social;
fprintf("\nH1: best LSTM (%s) vs CV: ADE %.1f%% lower (target >= 15%%) -> %s\n", ...
    bestName, 100*h1, passfail(h1 >= 0.15));
fprintf("H2: near robot, LSTM-robot vs LSTM-social: ADE %.1f%% lower, better in %d of %d test runs -> %s\n", ...
    100*h2, sum(h2runs), numel(h2runs), passfail(h2 > 0 && all(h2runs)));

%% Figures
f = figure("Visible","off","Position",[100 100 900 380]); lightTheme(f);
tiledlayout(1,2);
nexttile; bar(categorical(T.model, T.model), [T.ADE_m T.FDE_m]); ylabel("error (m)");
legend("ADE","FDE","Location","northwest"); title("All test snippets (17 May)");
nexttile; bar(categorical(T.model, T.model), [T.ADE_near_robot_m T.FDE_near_robot_m]); ylabel("error (m)");
legend("ADE","FDE","Location","northwest"); title("Robot within 3 m");
exportgraphics(f, fullfile(OUT, "ade_fde.png"), "Resolution", 150);

f = figure("Visible","off","Position",[100 100 1000 600]); lightTheme(f);
tiledlayout(2,3);
idx = find(near); rng(0); idx = idx(randperm(numel(idx), min(6, numel(idx))));
Pb = mean(preds.(matlab.lang.makeValidName(bestName)), 4);
for i = idx'
    nexttile; hold on; axis equal; grid on;
    plot(D.test_ego(i,:,1), D.test_ego(i,:,2), "k.-");
    plot(squeeze(D.test_fut(i,:,1)), squeeze(D.test_fut(i,:,2)), "g.-");
    plot(squeeze(predCV(i,:,1)), squeeze(predCV(i,:,2)), "b--");
    plot(squeeze(Pb(i,:,1)), squeeze(Pb(i,:,2)), "r.-");
    rb = squeeze(D.test_rob(i,:,:)); rb = rb(rb(:,5) > 0, :);
    if ~isempty(rb), plot(rb(:,1), rb(:,2), "s-", "Color", [1 .5 0]); end
end
legend(["past","true future","constant velocity",bestName,"robot"], "Location","bestoutside");
exportgraphics(f, fullfile(OUT, "examples.png"), "Resolution", 150);

%% Save models
save(fullfile(OUT, "results.mat"), "T", "R", "h1", "h2", "bestName", "nets", "-v7");
best = nets.(matlab.lang.makeValidName(bestName));
save(fullfile(OUT, "predictor_" + strrep(bestName,"-","_") + ".mat"), "best", "-v7");
% plain weight arrays for every variant, so the ROS 2 node can run the LSTM in
% numpy without MATLAB or an ONNX runtime
for v = variants
    W = plainWeights(nets.(matlab.lang.makeValidName(v))); %#ok<NASGU>
    save(fullfile(OUT, "weights_" + strrep(v,"-","_") + ".mat"), "-struct", "W", "-v7");
end
robotNet = nets.LSTM_robot;
try
    exportONNXNetwork(robotNet, fullfile(OUT, "predictor_LSTM_robot.onnx"));
    exportONNXNetwork(best, fullfile(OUT, "predictor_" + strrep(bestName,"-","_") + ".onnx"));
    fprintf("ONNX models exported.\n");
catch err
    fprintf("ONNX export skipped (%s). The .mat models are saved.\n", err.message);
end
fprintf("Done. Results in %s/\n", OUT);

%% ---------------------------------------------------------------- helpers
function D = loadData(file)
    % one .mat, or the same data split into thor_magni_sc3_meta.mat + _partNN.mat
    % (small enough to upload through a browser)
    if isfile(file), D = load(file); return; end
    parts = dir("thor_magni_sc3_part*.mat");
    if isempty(parts) || ~isfile("thor_magni_sc3_meta.mat")
        error("Put %s (or the thor_magni_sc3_meta/_part files) next to this script.", file);
    end
    D = load("thor_magni_sc3_meta.mat");
    for p = sort(string({parts.name}))
        S = load(p);
        for f = string(fieldnames(S))'
            if isfield(D, f), D.(f) = cat(1, D.(f), S.(f)(:,:,:,:)); else, D.(f) = S.(f); end
        end
    end
    for s = ["train","val","test"], D.(s + "_near") = D.(s + "_near")(:); end
end

function X = features(D, split, variant)
    ego = D.(split + "_ego");                       % N x 8 x 4
    F = ego;
    if variant ~= "LSTM-ego"
        nbr = D.(split + "_nbr");                   % N x 8 x 4 x 5
        F = cat(3, F, reshape(nbr, size(nbr,1), size(nbr,2), []));
    end
    if variant == "LSTM-robot"
        F = cat(3, F, D.(split + "_rob"));          % N x 8 x 5
    end
    F = permute(F, [2 3 1]);                        % time x features x N (trainnet: t-by-c)
    X = squeeze(num2cell(F, [1 2]));
end

function Y = flatFuture(fut)                        % N x 12 x 2 -> N x 24
    Y = reshape(fut, size(fut,1), []);
end

function P = unflatFuture(Y, nFut)
    P = reshape(Y, size(Y,1), nFut, 2);
end

function layers = buildNet(nFeat, nFut)
    layers = [
        sequenceInputLayer(nFeat, "Normalization", "zscore")
        lstmLayer(128, "OutputMode", "last")
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(2*nFut)];
end

function o = options(Xva, Yva, epochs)
    o = trainingOptions("adam", ...
        "MaxEpochs", epochs, "MiniBatchSize", 512, ...
        "InitialLearnRate", 2e-3, "LearnRateSchedule", "piecewise", ...
        "LearnRateDropPeriod", 10, "LearnRateDropFactor", 0.5, ...
        "Shuffle", "every-epoch", "ValidationData", {Xva, Yva}, ...
        "ValidationFrequency", 70, "OutputNetwork", "best-validation", ...
        "Plots", "none", "Verbose", true, "VerboseFrequency", 70);
end

function P = constantVelocity(ego, nFut, dt)
    v = ego(:, end, 3:4);                           % last observed velocity (local frame)
    p = ego(:, end, 1:2);                           % = origin
    k = reshape(1:nFut, 1, nFut);
    P = p + v .* (k * dt);
end

function [ade, fde] = errors(P, fut)
    d = sqrt(sum((P - fut).^2, 3));                 % N x 12
    ade = mean(d, 2); fde = d(:, end);
end

function idx = runIndex(D, testRuns)
    % snippets were concatenated run by run; recover the run of each snippet
    counts = zeros(numel(testRuns), 1);
    if isfield(D, "test_run_counts"), counts = double(D.test_run_counts(:)); end
    if all(counts == 0), idx = ones(size(D.test_fut,1), 1); return; end
    idx = repelem((1:numel(counts))', counts);
end

function D = subsetSplit(D, split, n)
    n = min(n, size(D.(split + "_fut"), 1));
    for k = ["ego","nbr","rob","fut","near"]
        A = D.(split + "_" + k);
        if isvector(A), D.(split + "_" + k) = A(1:n); else, D.(split + "_" + k) = A(1:n,:,:,:); end
    end
    if isfield(D, split + "_run_counts")
        c = double(D.(split + "_run_counts")); c = min(c, max(0, n - [0 cumsum(c(1:end-1))]));
        D.(split + "_run_counts") = c;
    end
end

function W = plainWeights(net)
    % LSTM gate order in MATLAB: input, forget, cell candidate, output
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
    W.notes = "x: time x features, z-scored with in_mean/in_std; LSTM gates i,f,g,o; " + ...
              "out = fc2(relu(fc1(h_last))) -> 24 = [x1..x12, y1..y12] in the person frame (m).";
end

function lightTheme(f)
    try, f.Theme = "light"; catch, set(f, "Color", "w"); end
end

function s = passfail(ok)
    if ok, s = "SUPPORTED"; else, s = "NOT SUPPORTED"; end
end
