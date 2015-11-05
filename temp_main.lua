require 'nn'
require 'gnuplot'
require 'optim'
require 'nngraph'
require 'tools'
require 'vis'

require 'utils'
require 'OneHot'
local CharSplitLMMinibatchLoader = require 'CharSplitLMMinibatchLoader'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')

-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'cf', 'cf or lstm')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',5,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',50,'number of timesteps to unroll for')

cmd:option('-steps_per_output',1,'number of feedback steps to run per output')
cmd:option('-num_functions',65,'number of function layers to create')

cmd:option('-controller_nonlinearity','tanh','nonlinearity for output of controller. Sets the range of the weights.')
cmd:option('-function_nonlinearity','sigmoid','nonlinearity for functions. sets range of function output')
-- cmd:option('-num_functions',65,'number of function layers to create')



cmd:option('-batch_size',30,'number of sequences to train on in parallel')

cmd:option('-max_epochs',20,'number of full passes through the training data')
cmd:option('-grad_clip',3,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-import', '', 'initialize network parameters from checkpoint at this path')


-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
-- cmd:option('-eval_val_every',10,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'networks', 'output directory where checkpoints get written')
cmd:option('-name','net','filename to autosave the checkpont to. Will be inside checkpoint_dir/')

-- GPU/CPU
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

local savedir = string.format('%s/%s', opt.checkpoint_dir, opt.name)
os.execute('mkdir -p '..savedir)
os.execute(string.format('rm %s/*', savedir))

-- log out the options used for creating this network to a file in the save directory.
-- super useful when you're moving folders around so you don't lose track of things.
local f = io.open(savedir .. '/opt.txt', 'w')
for key, val in pairs(opt) do
  f:write(tostring(key) .. ": " .. tostring(val) .. "\n")
end
f:flush()
f:close()

if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
end

local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac}

-- create the data loader class
local loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
local vocab_size = loader.vocab_size  -- the number of distinct characters
local vocab = loader.vocab_mapping
print('vocab size: ' .. vocab_size)

local params, grad_params
-- if opt.import ~= '' then
--     checkpoint = torch.load(opt.import)
--     model = checkpoint.model
--     params, grad_params = model:getParameters()
--
--     local vocab_compatible = true
--     for c,i in pairs(checkpoint.vocab) do
--         if not vocab[c] == i then
--             vocab_compatible = false
--         end
--     end
--     assert(vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
--     -- overwrite model settings based on checkpoint to ensure compatibility
--     print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. checkpoint.opt.steps_per_output .. ', steps_per_output=' .. checkpoint.opt.num_functions .. ', num_functions=' .. ' based on the checkpoint.')
--     opt.rnn_size = checkpoint.opt.rnn_size
--     opt.num_layers = checkpoint.opt.num_layers
--     opt.steps_per_output = checkpoint.opt.steps_per_output
--     opt.num_functions = checkpoint.opt.num_functions
--
-- else
--     if opt.model == 'cf' then
--         require 'CFNetwork_multistep'
--         model = nn.CFNetwork({
--                 input_dimension = vocab_size,
--                 num_functions = opt.num_functions,
--                 controller_units_per_layer = opt.rnn_size,
--                 controller_num_layers = opt.num_layers,
--                 controller_dropout = opt.dropout,
--                 steps_per_output = opt.steps_per_output,
--                 controller_nonlinearity = opt.controller_nonlinearity,
--                 function_nonlinearity = opt.function_nonlinearity,
--             })
--     elseif opt.model == 'lstm' then
--         require 'SteppableLSTM'
--         model = nn.SteppableLSTM(vocab_size, vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
--     else
--         error("Model type not valid.")
--     end
--
--     params, grad_params = model:getParameters()
--     params:uniform(-0.08, 0.08) -- small numbers uniform
-- end

require 'CFNetwork_multistep'
cf = nn.CFNetwork({
        input_dimension = vocab_size,
        encoded_dimension = 33,
        num_functions = opt.num_functions,
        controller_units_per_layer = opt.rnn_size,
        controller_num_layers = opt.num_layers,
        controller_dropout = opt.dropout,
        steps_per_output = opt.steps_per_output,
        controller_nonlinearity = opt.controller_nonlinearity,
        function_nonlinearity = opt.function_nonlinearity,
    })

-- require 'SteppableLSTM'
-- cf = nn.SteppableLSTM(vocab_size, vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)

-- criterion = nn.Identity()
criterion = nn.CrossEntropyCriterion()
one_hot = OneHot(vocab_size)

-- model = nn.Sequential()
local model = nn.Sequential()

model:add(one_hot)
-- model:add(nn.Linear(vocab_size, 33))
model:add(cf)

-- hotswap_decoder = nn.Sequential()
-- hotswap_decoder:add(nn.Linear(33, vocab_size))
-- hotswap_decoder:add(nn.Tanh())
-- hotswap_decoder:add(nn.Linear(vocab_size, vocab_size))
-- cf.decoder = hotswap_decoder

-- hotswap_decoder = nn.Sequential()
-- hotswap_decoder:add(nn.Tanh())
-- hotswap_decoder:add(nn.Linear(33, vocab_size))
-- cf.decoder = hotswap_decoder

-- hotswap_decoder = nn.Sequential()
-- hotswap_decoder:add(nn.Linear(33, vocab_size))
-- cf.decoder = hotswap_decoder

hotswap_decoder = nn.Sequential()
hotswap_decoder:add(nn.Linear(33, vocab_size))
hotswap_decoder:add(nn.Tanh())
cf.decoder = hotswap_decoder


-- model:add(nn.Tanh())
-- model:add(nn.Linear(vocab_size, vocab_size))


-- main_model:add(nn.JoinTable(2))

-- local parallel = nn.ParallelTable()
-- parallel:add(main_model)
-- parallel:add(nn.Identity())

-- model:add(parallel)
-- model:add(criterion)

params, grad_params = model:getParameters()
params:uniform(-0.08, 0.08) -- small numbers uniform

if opt.gpuid >= 0 then
    model:cuda()
    -- criterion:cuda()
    -- one_hot:cuda()
end

-- if model.functions[1].modules[1].weight:type() == "torch.CudaTensor" then
--     criterion:cuda()
--     one_hot:cuda()
-- end


-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    cf:reset()
    model:evaluate()

    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
        if opt.gpuid >= 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            y = y:float():cuda()
        end
        -- forward pass
        for t=1,opt.seq_length do
            local input = one_hot:forward(x[{{}, t}])

            local prediction = model:step(input)
            -- print("Input:", vis.simplestr(input[1]))
            -- print("Prediction:", vis.simplestr(prediction[1]))
            loss = loss + criterion:forward(prediction, y[{{}, t}])
        end
        print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n
    return loss
end

-- profiler = xlua.Profiler('on', true)
-- do fwd/bwd and return loss, grad_params
function feval(x)
    -- profiler:start('batch')
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()
    cf:reset()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end

    ------------------- forward pass -------------------
    local predictions = {}           -- softmax outputs
    local grad_outputs = {}
    local inputs = {}
    local loss = 0

    model:training() -- make sure we are in correct mode (this is cheap, sets flag)

    predictions = model:forward(x)
    for t = 1, opt.seq_length do
        loss = loss + criterion:forward(predictions[t], y[{{}, t}])
        grad_outputs[t] = criterion:backward(predictions[t], y[{{}, t}]):clone()
    end

    -- loss = model:forward{x, y}
    -- print(loss)

    -- print(main_model)
    -- loss = main_model:forward(x)

    -- for t=1,opt.seq_length do
    --     inputs[t] = one_hot:forward(x[{{}, t}])
    --
    --     predictions[t] = model:step(inputs[t]) --:clone()
    --     -- if t == 4 and opt.model == 'cf' then
    --     --     vis.hist(model.controller.output[1])
    --     -- end
    --
    --     loss = loss + criterion:forward(predictions[t], y[{{}, t}])
    --
    --     grad_outputs[t] = criterion:backward(predictions[t], y[{{}, t}]):clone()
    --
    --
    --     -- print("pred:", predictions[t][1])
    --     -- print("truth:", y[{{}, t}][1])
    --
    --     -- print(grad_outputs[t][1])
    -- end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------

    model:backward(x, grad_outputs)
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    -- grad_params:mul(-1)
    -- profiler:lap('batch')
    collectgarbage()
    return loss, grad_params
end

-- [[
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil

for i = 1, iterations do
    local epoch = i / loader.ntrain

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    local time = timer:time().real

    -- profiler:printAll()

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss
        print(string.format('[epoch %.3f] Validation loss: %6.8f', epoch, val_loss))



        local model_file = string.format('%s/epoch%.2f_%.4f.t7', savedir, epoch, val_loss)
        print('saving checkpoint to ' .. model_file)
        local checkpoint = {}
        checkpoint.model = model
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(model_file, checkpoint)



        local val_loss_log = io.open(savedir ..'/val_loss.txt', 'a')
        val_loss_log:write(val_loss .. "\n")
        val_loss_log:flush()
        val_loss_log:close()
        -- os.execute("say 'Checkpoint saved.'")
        -- os.execute(string.format("say 'Epoch %.2f'", epoch))
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then
        loss0 = loss[1]
    end
    -- if loss[1] > loss0 * 8 then
    --     print('loss is exploding, aborting.')
    --     print("loss0:", loss0, "loss[1]:", loss[1])
    --     break -- halt
    -- end
end
