
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
cmd:option('-single_primitive_data','data/rotate.json','one-primitive dataset')
cmd:option('-full_data','data/primitives.json','dataset')

-- optimization
cmd:option('-learning_rate',2e-2,'learning rate')

cmd:option('-batch_size',1,'number of sequences to train on in parallel')

cmd:option('-max_epochs',20,'number of full passes through the training data')
cmd:option('-grad_clip',3,'clip gradients at this value')
cmd:option('-train_frac',0.9,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)

cmd:option('-import', '', 'test this network')


-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',100,'every how many iterations should we evaluate on validation data?')
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
    local name = 'forgetting_'
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

-- create the data loader for the one-primitive data
single_primitive_loader = ProgramBatchLoader.create(opt.single_primitive_data, opt.batch_size, split_sizes)
full_loader = ProgramBatchLoader.create(opt.full_data, opt.batch_size, split_sizes)

require 'IIDCF_meta'
checkpoint = torch.load(opt.import)
model = checkpoint.model

is_cf = model.functions ~= nil

if is_cf then
    controller_params, controller_grad_params = model:getControllerParameters()
    function_params, function_grad_params = model:getFunctionParameters()
else
    params, grad_params = model:getParameters()
end

criterion = nn.MSECriterion()
one_hot = OneHot(8)

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = full_loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    full_loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    if is_cf then
        model:reset()
    end
    model:evaluate()
    -- print("norm of model params at beginning of eval_split: ", params:norm())

    val_count = 0
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = full_loader:next_batch(split_index)
        if opt.gpuid >= 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            y = y:float():cuda()
        end

        -- only test those that are not the rotate primitive
        -- print('x[1][1]: ', x[1][1])
        if x[1][1] ~= 0 then
            if is_cf then
                model:reset()
                primitive = one_hot:forward(x[1])
                input = {primitive, x[2]}
            else
                primitive = one_hot:forward(x[1])
                input = torch.zeros(1, 18)
                input[{1, {1, 8}}] = primitive:clone()
                input[{1, {9, 18}}] = x[2]:clone()
            end

            output = model:forward(input)
            step_loss = criterion:forward(output, y)

            local primitive_index = x[1][1]
            if is_cf then
                print("Primitive: " .. tostring(primitive_index).. " Loss: " .. tostring(step_loss) .. " Weights: " .. vis.simplestr(model.controller.output[1]))
            else
                -- print("norm of model params after an eval_split: ", params:norm())
                -- print("Primitive: " .. tostring(primitive_index).. " Loss: " .. tostring(step_loss))
            end

            if i % 100 == 0 then
                -- print(vis.simplestr(output[1]))
                -- print(vis.simplestr(y[1]))
            end

            loss = loss + step_loss
            val_count = val_count + 1
        end
    end

    loss = loss / val_count
    return loss
end

-- profiler = xlua.Profiler('on', true)
-- do fwd/bwd and return loss, grad_params
function feval(x)
    -- profiler:start('batch')
    if x ~= params then
        error("Params not equal to given feval argument.")
        print("Params not equal to given feval argument.")
        params:copy(x)
    end
    -- print("norm of model params at beginning of feval: ", params:norm())
    if is_cf then
        controller_grad_params:zero()
        function_grad_params:zero()
        model:reset()
    else
        grad_params:zero()
    end

    ------------------ get minibatch -------------------
    local x, y = single_primitive_loader:next_batch(1)
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end

    ------------------- forward pass -------------------
    model:evaluate() -- make sure we are in correct mode
    -- print("norm of model params after setting model to evaluate: ", params:norm())

    -- print(input)
    -- print("primitive onehot: ", primitive)
    -- print("input: ", x[2])
    -- print("target: ", y)

    local loss
    if is_cf then
        local primitive_index = x[1][1]
        print("Primitive:", primitive_index)

        primitive = one_hot:forward(x[1])
        input = {primitive, x[2]}

        output = model:forward(input)

        loss = criterion:forward(output, y)
        grad_output = criterion:backward(output, y):clone()

        ------------------ backward pass -------------------
        model:backward(input, grad_output)
        controller_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
        function_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    else
        local primitive_index = x[1][1]
        print("Primitive:", primitive_index)
        local input, primitive

        primitive = one_hot:forward(x[1])
        input = torch.zeros(1, 18)
        input[{1, {1, 8}}] = primitive:clone()
        input[{1, {9, 18}}] = x[2]:clone()

        output = model:forward(input)

        loss = criterion:forward(output, y)

        -- print("norm of model params after feval forward: ", params:norm())

        grad_output = criterion:backward(output, y):clone()

        ------------------ backward pass -------------------
        model:backward(input, grad_output)

        grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    end

    print(vis.simplestr(output[1]))
    print(vis.simplestr(y[1]))

    -- print("norm of model params at end of feval: ", params:norm())

    collectgarbage()
    return loss, grad_params
end

-- [[
train_losses = {}
val_losses = {}

local controller_optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local function_optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}

local iterations = opt.max_epochs * single_primitive_loader.ntrain
local iterations_per_epoch = single_primitive_loader.ntrain
local loss0 = nil

-- large number for ScheduledWeightSharpener
iteration = 10000000000

for step = 1, iterations do
    print('')
    -- print("norm of model params at beginning of step: ", params:norm())
    epoch = step / single_primitive_loader.ntrain

    local timer = torch.Timer()

    if is_cf then
        loss, _ = feval(params)
        function feval_controller()
            return loss, controller_grad_params
        end
        function feval_function()
            return loss, function_grad_params
        end

        local _, loss_temp = optim.rmsprop(feval_controller, controller_params, controller_optim_state)
        local _, loss_temp = optim.rmsprop(feval_function, function_params, function_optim_state)
        loss = loss_temp

        local time = timer:time().real

        -- profiler:printAll()

        train_loss = loss[1] -- the loss is inside a list, pop it
        train_losses[step] = train_loss

        -- print(step, ' : ', iterations, ' : ', epoch, ' : ', train_loss, ' : ', controller_grad_params:norm() / controller_params:norm(), ' : ', function_grad_params:norm() / function_params:norm(), ' : ', time)
        if step % opt.print_every == 0 then
            print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, controller grad/param norm = %6.4e, function grad/param norm = %6.4e, time/batch = %.2fs", step, iterations, epoch, train_loss, controller_grad_params:norm() / controller_params:norm(), function_grad_params:norm() / function_params:norm(), time))
        end
    else
        local _, loss_temp = optim.rmsprop(feval, params, optim_state)
        loss = loss_temp

        local time = timer:time().real

        train_loss = loss[1] -- the loss is inside a list, pop it
        train_losses[step] = train_loss

        if step % opt.print_every == 0 then
            print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", step, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
        end
    end

    -- print("norm of model params after update: ", params:norm())

    -- every now and then or on last iteration
    if step % opt.eval_val_every == 0 or step == iterations or step == 1 then
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
