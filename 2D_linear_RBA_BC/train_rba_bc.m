function result = train_rba_bc(varargin)
%TRAIN_RBA_BC Compatibility wrapper for the GPU paper-training entrypoint.
result = train_rba_bc_gpu(varargin{:});
end
