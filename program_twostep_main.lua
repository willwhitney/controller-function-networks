
require 'nn'
require 'gnuplot'
require 'optim'
require 'nngraph'
require 'distributions'

require 'vis'

require 'utils'
require 'OneHot'
require 'ExpectationCriterion'
local ProgramBatchLoader = require 'ProgramBatchLoader'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')

-- data
cmd:option('-data_file','data/twostep.json','dataset')
cmd:option('-num_primitives',8,'how many primitives are in this data')

-- model params
cmd:option('-rnn_size', 10, 'size of LSTM internal state')
cmd:option('-layer_size', 10, 'size of the layers')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-model', 'scheduled_sharpening', 'cf or sampling')
cmd:option('-metadata_only_controller', false, 'determines whether controller should get metadata and input, or only metadata')
cmd:option('-all_metadata_controller', false, 'determines whether controller gets all metadata at once instead of one step at a time')


cmd:option('-criterion', 'L2', 'L2 or L1') -- used only for sampling


cmd:option('-sharpening_rate', 10, 'the slope (per 10K iterations) for the sharpening exponent')


-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-function_learning_rate',1e-2,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',15000,'in number of examples, when to start decaying the learning rate')
cmd:option('-learning_rate_decay_interval',2000,'in number of examples, how often to decay the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-noise',0,'variance of noise added to the weights before sharpening')

-- cmd:option('-seq_length',50,'number of timesteps to unroll for')

cmd:option('-steps_per_output',2,'number of feedback steps to run per output')
cmd:option('-num_functions',8,'number of function layers to create')

cmd:option('-controller_nonlinearity','softmax','nonlinearity for output of controller. Sets the range of the weights.')
cmd:option('-function_nonlinearity','prelu','nonlinearity for functions. sets range of function output')
-- cmd:option('-num_functions',65,'number of function layers to create')



cmd:option('-batch_size',1,'number of sequences to train on in parallel')

cmd:option('-max_epochs',20,'number of full passes through the training data')
cmd:option('-grad_clip',3,'clip gradients at this value')
cmd:option('-train_frac',0.9,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-import', '', 'initialize network parameters from checkpoint at this path')


-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',2000,'every how many iterations should we evaluate on validation data?')
-- cmd:option('-eval_val_every',10,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'networks', 'output directory where checkpoints get written')
cmd:option('-name','net','filename to autosave the checkpont to. Will be inside checkpoint_dir/')

-- GPU/CPU
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

if opt.name == 'net' then
    local name = 'twostep_'
    for k, v in ipairs(arg) do
        name = name .. string.gsub(tostring(v), '/', '.') .. '_'
    end
    opt.name = name .. os.date("%b_%d_%H_%M_%S")
end

local savedir = string.format('%s/%s', opt.checkpoint_dir, opt.name)
print("Saving output to "..savedir)
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

local logfile = io.open(savedir .. '/output.log', 'w')
true_print = print
print = function(...)
    for i, v in ipairs{...} do
        true_print(v)
        logfile:write(tostring(v))
    end
    logfile:write("\n")
    logfile:flush()
end

if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
end

local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac}

-- create the data loader class
loader = ProgramBatchLoader.create(opt.data_file, opt.batch_size, split_sizes)

if opt.model == 'cf' then
    require 'IIDCF_meta'
    model = nn.IIDCFNetwork({
            num_primitives = opt.num_primitives,
            encoded_dimension = 10,
            num_functions = opt.num_functions,
            controller_units_per_layer = opt.rnn_size,
            controller_num_layers = opt.num_layers,
            controller_dropout = opt.dropout,
            steps_per_output = opt.steps_per_output,
            controller_nonlinearity = opt.controller_nonlinearity,
            function_nonlinearity = opt.function_nonlinearity,
            controller_type = 'normal',
        })
elseif opt.model == 'sharpening' then
    require 'IIDCF_meta'
    model = nn.IIDCFNetwork({
            num_primitives = opt.num_primitives,
            encoded_dimension = 10,
            num_functions = opt.num_functions,
            controller_units_per_layer = opt.rnn_size,
            controller_num_layers = opt.num_layers,
            controller_dropout = opt.dropout,
            steps_per_output = opt.steps_per_output,
            controller_nonlinearity = opt.controller_nonlinearity,
            function_nonlinearity = opt.function_nonlinearity,
            controller_type = 'sharpening',
        })
elseif opt.model == 'scheduled_sharpening' then
    require 'IIDCF_meta'
    model = nn.IIDCFNetwork({
            num_primitives = opt.num_primitives,
            encoded_dimension = 10,
            num_functions = opt.num_functions,
            controller_units_per_layer = opt.rnn_size,
            controller_num_layers = opt.num_layers,
            controller_dropout = opt.dropout,
            steps_per_output = opt.steps_per_output,
            controller_nonlinearity = opt.controller_nonlinearity,
            function_nonlinearity = opt.function_nonlinearity,
            controller_type = 'scheduled_sharpening',
            controller_noise = opt.noise,
            all_metadata_controller = opt.all_metadata_controller,
            metadata_only_controller = opt.metadata_only_controller,
        })
elseif opt.model == 'ff-controller' then
    require 'FF_IIDCF_meta'
    model = nn.IIDCFNetwork({
            input_dimension = opt.num_primitives + 10,
            encoded_dimension = 10,
            num_functions = opt.num_functions,
            controller_units_per_layer = opt.rnn_size,
            controller_num_layers = opt.num_layers,
            controller_dropout = opt.dropout,
            steps_per_output = opt.steps_per_output,
            function_nonlinearity = opt.function_nonlinearity,
            controller_noise = opt.noise,
        })
elseif opt.model == 'sampling' then
    require 'SamplingIID'
    model = nn.SamplingCFNetwork({
            input_dimension = opt.num_primitives + 10,
            encoded_dimension = 10,
            num_functions = opt.num_functions,
            controller_units_per_layer = opt.rnn_size,
            controller_num_layers = opt.num_layers,
            controller_dropout = opt.dropout,
            steps_per_output = opt.steps_per_output,
            controller_nonlinearity = opt.controller_nonlinearity,
            function_nonlinearity = opt.function_nonlinearity,
            criterion = opt.criterion,
        })
else
    error("Model type not valid.")
end

-- put the pretrained functions from the loaded model into the new model
if opt.import ~= '' then
    checkpoint = torch.load(opt.import)
    model.functions = checkpoint.model.functions
end

controller_params, controller_grad_params = model:getControllerParameters()
function_params, function_grad_params = model:getFunctionParameters()
controller_params:uniform(-0.9, 0.9) -- small numbers uniform
function_params:uniform(0.0, 0.2) -- small numbers uniform


criterion = nn.MSECriterion()
one_hot = OneHot(opt.num_primitives)

-- [[

if opt.gpuid >= 0 then
    model:cuda()
    criterion:cuda()
    one_hot:cuda()
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
    model:reset()
    model:evaluate()

    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
        if opt.gpuid >= 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            y = y:float():cuda()
        end

        model:reset()
        local primitive_index = x[1][1]
        local input, output, primitive
        primitive = one_hot:forward(x[1][1])
        input = {primitive, x[2]}

        local step_loss = 0

        oldprint = print
        print = function() end
        if opt.model == 'sampling' then
            -- input = {primitive, x[2]}
            -- print(input)
            step_loss = model:forward(input, y)
            local probabilities, outputs = table.unpack(model.output_value)
            output = torch.zeros(outputs[1]:size())

            for output_index = 1, outputs:size(1) do
                output = output + outputs[output_index] * probabilities[1][output_index]
            end
            output = output:reshape(1, output:size(1))
        else
            -- primitive = one_hot:forward(x[1])
            -- input = {primitive, x[2]}
            output = model:forward(input)
            step_loss = criterion:forward(output, y)
        end
        print = oldprint

        if i % 100 == 0 then
            print("Primitive: ", primitive_index, " Loss: ", step_loss,
                    " Weights: ", vis.simplestr(model.controller.output[1]))
            print(vis.simplestr(output[1]))
            print(vis.simplestr(y[1]))
        end

        loss = loss + step_loss
    end

    loss = loss / n
    return loss
end

-- profiler = xlua.Profiler('on', true)
-- do fwd/bwd and return loss, grad_params
function feval(x)
    -- profiler:start('batch')
    if x ~= params then
        error("Params not equal to given feval argument.")
        params:copy(x)
    end
    -- grad_params:zero()
    controller_grad_params:zero()
    function_grad_params:zero()
    model:reset()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    -- print("x: ", x)
    -- print("x1: ", x[1])
    -- print("x2: ", x[2])

    ------------------- forward pass -------------------
    model:training() -- make sure we are in correct mode

    local primitive_index = x[1][1]:clone()
    primitive_index[1] = x[1][1][2]
    primitive_index[2] = x[1][1][1]
    print("Primitive:", primitive_index)
    local input, output, primitive, loss

    primitive = one_hot:forward(x[1][1])
    input = {primitive, x[2]}


    if opt.model == 'sampling' then

        -- print(input)
        loss = model:forward(input, y)
        local probabilities, outputs = table.unpack(model.output_value)
        output = torch.zeros(outputs[1]:size())
        print(outputs)
        for i = 1, outputs:size(1) do
            output = output + outputs[i] * probabilities[1][i]
        end
        output = output:reshape(1, output:size(1))

        ------------------ backward pass -------------------
        model:backward(input, y)
    else
        output = model:forward(input)
        loss = criterion:forward(output, y)
        grad_output = criterion:backward(output, y):clone()

        ------------------ backward pass -------------------
        model:backward(input, grad_output)
    end
    controller_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    function_grad_params:clamp(-opt.grad_clip, opt.grad_clip)

    print(vis.simplestr(output[1]))
    print(vis.simplestr(y[1]))

    collectgarbage()
    return loss, grad_params
end

-- [[
train_losses = {}
val_losses = {}
local controller_optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local function_optim_state = {learningRate = opt.function_learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil

for step = 1, iterations do
    iteration = step
    print('')
    epoch = step / loader.ntrain

    local timer = torch.Timer()

    local loss, _ = feval(params)
    function feval_controller()
        return loss, controller_grad_params
    end
    function feval_function()
        return loss, function_grad_params
    end

    local _, loss = optim.rmsprop(feval_controller, controller_params, controller_optim_state)
    local _, loss = optim.rmsprop(feval_function, function_params, function_optim_state)

    local time = timer:time().real

    -- profiler:printAll()

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[step] = train_loss

    -- exponential learning rate decay
    if step % opt.learning_rate_decay_interval == 0 and opt.learning_rate_decay < 1 then
        if step >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            controller_optim_state.learningRate = controller_optim_state.learningRate * decay_factor -- decay it
            function_optim_state.learningRate = function_optim_state.learningRate * decay_factor -- decay it
            print('decayed controller learning rate by a factor ' .. decay_factor .. ' to ' .. controller_optim_state.learningRate)
            print('decayed function learning rate by a factor ' .. decay_factor .. ' to ' .. function_optim_state.learningRate)
        end
    end

    if step % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, controller grad/param norm = %6.4e, function grad/param norm = %6.4e, time/batch = %.2fs", step, iterations, epoch, train_loss, controller_grad_params:norm() / controller_params:norm(), function_grad_params:norm() / function_params:norm(), time))
    end

    -- every now and then or on last iteration
    if step % opt.eval_val_every == 0 or step == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[step] = val_loss
        print(string.format('[epoch %.3f] Validation loss: %6.8f', epoch, val_loss))



        local model_file = string.format('%s/epoch%.2f_%.4f.t7', savedir, epoch, val_loss)
        print('saving checkpoint to ' .. model_file)
        local checkpoint = {}
        checkpoint.model = model
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.step = step
        checkpoint.epoch = epoch
        torch.save(model_file, checkpoint)



        local val_loss_log = io.open(savedir ..'/val_loss.txt', 'a')
        val_loss_log:write(val_loss .. "\n")
        val_loss_log:flush()
        val_loss_log:close()
        -- os.execute("say 'Checkpoint saved.'")
        -- os.execute(string.format("say 'Epoch %.2f'", epoch))
    end

    if step % 10 == 0 then collectgarbage() end

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
-- --]]
