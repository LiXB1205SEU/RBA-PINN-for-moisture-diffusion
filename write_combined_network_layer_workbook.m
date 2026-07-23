function outputFile = write_combined_network_layer_workbook(opts)
%WRITE_COMBINED_NETWORK_LAYER_WORKBOOK Collect all network-layer results in one workbook.
%
% Default cases:
%   1D_linear_BC, 1D_nonlinear_BC, 1D_selfdrying_BC, 2D_selfdrying_BC

if nargin < 1 || isempty(opts)
    opts = struct();
end

rootDir = fileparts(mfilename('fullpath'));

D.resultsRoot = fullfile(rootDir, 'results');
D.caseNames = ["1D_linear_BC", "1D_nonlinear_BC", "1D_selfdrying_BC", "2D_selfdrying_BC"];
D.outputFile = fullfile(D.resultsRoot, 'network_layer_all_cases_summary.xlsx');

opts = applyDefaults(opts, D);
ensureDir(opts.resultsRoot);

summaryTotal = readAndStack(opts.resultsRoot, opts.caseNames, 'summary_total_mean_variance.csv');
summaryByTime = readAndStack(opts.resultsRoot, opts.caseNames, 'summary_by_time_mean_variance.csv');
allFoldsTotal = readAndStack(opts.resultsRoot, opts.caseNames, 'all_folds_total.csv');
allFoldsByTime = readAndStack(opts.resultsRoot, opts.caseNames, 'all_folds_by_time.csv');
bestByCase = makeBestByCase(summaryTotal, opts.caseNames);

outputFile = opts.outputFile;
if isfile(outputFile)
    delete(outputFile);
end

writetable(summaryTotal, outputFile, 'Sheet', 'summary_total');
writetable(summaryByTime, outputFile, 'Sheet', 'summary_by_time');
writetable(allFoldsTotal, outputFile, 'Sheet', 'all_folds_total');
writetable(allFoldsByTime, outputFile, 'Sheet', 'all_folds_by_time');
writetable(bestByCase, outputFile, 'Sheet', 'best_by_case');

fprintf('Combined network-layer workbook written: %s\n', outputFile);
end

function out = readAndStack(resultsRoot, caseNames, fileName)
out = table();

for i = 1:numel(caseNames)
    caseName = caseNames(i);
    filePath = fullfile(resultsRoot, char(caseName), fileName);

    if ~isfile(filePath)
        warning('Missing network-layer result file: %s', filePath);
        continue;
    end

    T = readtable(filePath, 'TextType', 'string');
    out = appendTable(out, T);
end
end

function bestByCase = makeBestByCase(summaryTotal, caseNames)
bestByCase = table();

if isempty(summaryTotal)
    return;
end

caseColumn = string(summaryTotal.Case);
for i = 1:numel(caseNames)
    caseName = caseNames(i);
    idx = find(caseColumn == caseName);

    if isempty(idx)
        continue;
    end

    vals = summaryTotal.Mean_Relative_L2(idx);
    vals(isnan(vals)) = inf;
    [~, localIndex] = min(vals);
    bestByCase = appendTable(bestByCase, summaryTotal(idx(localIndex), :));
end
end

function opts = applyDefaults(opts, defaults)
names = fieldnames(defaults);
for i = 1:numel(names)
    name = names{i};
    if ~isfield(opts, name) || isempty(opts.(name))
        opts.(name) = defaults.(name);
    end
end
end

function ensureDir(folderPath)
if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end
end

function out = appendTable(out, rows)
if isempty(out)
    out = rows;
else
    out = [out; rows];
end
end
