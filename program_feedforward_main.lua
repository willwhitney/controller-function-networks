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

-- data
cmd:option('-data_file','data/primitives.json','dataset')
cmd:option('-num_primitives',8,'how many primitives are in this data')

-- optimization
cmd:option('-learning_rate',2e-2,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',15000,'in number of examples, when to start decaying the learning rate')
cmd:option('-learning_rate_decay_interval',2000,'in number of examples, how often to decay the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')

cmd:option('-batch_size',1,'number of sequences to train on in parallel')
cmd:option('-max_epochs',20,'number of full passes through the training data')
cmd:option('-grad_clip',3,'clip gradients at this value')
cmd:option('-train_frac',0.9,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')

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
    local name = 'feedforward_wide_'
    for k, v in ipairs(arg) do
        name = name .. tostring(v) .. '_'
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

local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac}

-- create the data loader class
loader = ProgramBatchLoader.create(opt.data_file, opt.batch_size, split_sizes)

model = nn.Sequential()
model:add(nn.Linear(18, 100))
model:add(nn.PReLU())
model:add(nn.Linear(100, 100))
model:add(nn.PReLU())
model:add(nn.Linear(100, 10))
model:add(nn.PReLU())

params, grad_params = model:getParameters()
params:uniform(0.0, 0.2) -- small numbers uniform

criterion = nn.MSECriterion()
one_hot = OneHot(opt.num_primitives)

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    model:evaluate()

    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
        if opt.gpuid >= 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            y = y:float():cuda()
        end

        local primitive_index = x[1][1]
        local input, output, primitive

        local step_loss = 0

        primitive = one_hot:forward(x[1])
        input = torch.zeros(1, 18)
        input[{1, {1, 8}}] = primitive:clone()
        input[{1, {9, 18}}] = x[2]:clone()

        output = model:forward(input)
        step_loss = criterion:forward(output, y)

        if i % 100 == 0 then
            print("Primitive: ", primitive_index, " Loss: ", step_loss)
            print(vis.simplestr(output[1]))
            print(vis.simplestr(y[1]))
        end

        loss = loss + step_loss
    end

    loss = loss / n
    return loss
end

-- do fwd/bwd and return loss, grad_params
function feval(x)
    -- if x ~= params then
    --     error("Params not equal to given feval argument.")
    --     params:copy(x)
    -- end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end

    ------------------- forward pass -------------------
    model:training() -- make sure we are in correct mode

    local primitive_index = x[1][1]
    print("Primitive:", primitive_index)
    local input, output, primitive, loss

    primitive = one_hot:forward(x[1])
    input = torch.zeros(1, 18)
    input[{1, {1, 8}}] = primitive:clone()
    input[{1, {9, 18}}] = x[2]:clone()

    output = model:forward(input)

    loss = criterion:forward(output, y)
    grad_output = criterion:backward(output, y):clone()

    ------------------ backward pass -------------------
    model:backward(input, grad_output)

    grad_params:clamp(-opt.grad_clip, opt.grad_clip)

    print(vis.simplestr(output[1]))
    print(vis.simplestr(y[1]))

    collectgarbage()
    return loss, grad_params
end

train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil

for step = 1, iterations do
    iteration = step
    print('')
    epoch = step / loader.ntrain

    local timer = torch.Timer()

    local _, loss = optim.rmsprop(feval, params, optim_state)

    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[step] = train_loss

    -- exponential learning rate decay
    if step % opt.learning_rate_decay_interval == 0 and opt.learning_rate_decay < 1 then
        if step >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed function learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    if step % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", step, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
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
--]]


model = nn.Sequential()
model:add(nn.Linear(10, 10))
model:add(nn.PReLU())
model:add(nn.Linear(10, 10))
model:add(nn.PReLU())
