-- Modified from https://github.com/karpathy/char-rnn, who got it from
-- https://github.com/oxford-cs-ml-2015/practical6

require 'json'

local ProgramBatchLoader = {}
ProgramBatchLoader.__index = ProgramBatchLoader

function ProgramBatchLoader.create(json_file, batch_size, split_fractions)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    local self = {}
    setmetatable(self, ProgramBatchLoader)

    self.batch_size = batch_size
    self.dataset = json.load(json_file)

    -- cut off the end so that it divides evenly
    local len = #self.dataset
    if len % batch_size ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        self.dataset = table.pack(table.unpack(self.dataset, 1, math.floor(len / batch_size)))
    end

    function build_next_batch(starting_index)
        -- print(starting_index)
        if starting_index + self.batch_size > #self.dataset then
            return nil
        else
            local primitive_indices = {}
            local inputs = {}
            local outputs = {}
            for i = 1, self.batch_size do
                table.insert(primitive_indices, self.dataset[starting_index + i][1])
                table.insert(inputs, self.dataset[starting_index + i][2])
                table.insert(outputs, self.dataset[starting_index + i][3])
            end
            local batch = {
                    torch.Tensor(primitive_indices),
                    torch.Tensor(inputs),
                    torch.Tensor(outputs),
                }
            return batch, starting_index + self.batch_size
        end
    end

    self.batches = {}
    local current_batch, next_index = build_next_batch(0)
    while current_batch ~= nil do
        table.insert(self.batches, current_batch)
        current_batch, next_index = build_next_batch(next_index)
    end

    self.num_batches = #self.batches

    -- lets try to be helpful here
    if self.num_batches < 50 then
        print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end

    -- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
    if split_fractions[3] == 0 then
        -- catch a common special case where the user might not want a test set
        self.ntrain = math.floor(self.num_batches * split_fractions[1])
        self.nval = self.num_batches - self.ntrain
        self.ntest = 0
    else
        -- divide data to train/val and allocate rest to test
        self.ntrain = math.floor(self.num_batches * split_fractions[1])
        self.nval = math.floor(self.num_batches * split_fractions[2])
        self.ntest = self.num_batches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
    end

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}


    local start_indices = {1, self.ntrain + 1, self.ntrain + self.nval + 1}
    self.dataset_split = {{}, {}, {}}
    for i, batch in ipairs(self.batches) do
        if i < start_indices[2] then
            table.insert(self.dataset_split[1], batch)
        elseif i < start_indices[3] then
            table.insert(self.dataset_split[2], batch)
        else
            table.insert(self.dataset_split[3], batch)
        end
    end

    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', #self.dataset_split[1], #self.dataset_split[2], #self.dataset_split[3]))
    collectgarbage()
    return self
end

function ProgramBatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function ProgramBatchLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end

    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    -- if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    -- if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
    local batch = self.dataset_split[split_index][ix]
    local x = {batch[1], batch[2]}
    local y = batch[3]
    return x, y
end

return ProgramBatchLoader
