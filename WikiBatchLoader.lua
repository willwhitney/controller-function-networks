
-- Modified from https://github.com/karpathy/char-rnn, who got it from
-- https://github.com/oxford-cs-ml-2015/practical6

local WikiBatchLoader = {}
WikiBatchLoader.__index = WikiBatchLoader

function WikiBatchLoader.create(preprocessed_data_file, split_fractions)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    local self = {}
    setmetatable(self, WikiBatchLoader)

    self.dataset = torch.load(preprocessed_data_file)

    self.num_batches = self.dataset:size(1)
    self.batch_size = self.dataset:size(2)
    self.seq_length = self.dataset:size(3)

    -- local ydata = data:clone()
    -- ydata:sub(1,-2):copy(data:sub(2,-1))
    -- ydata[-1] = data[1]
    -- self.x_batches = data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    -- self.nbatches = #self.x_batches
    -- self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    -- assert(#self.x_batches == #self.y_batches)
    --
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
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = self.nbatches - self.ntrain
        self.ntest = 0
    else
        -- divide data to train/val and allocate rest to test
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = math.floor(self.nbatches * split_fractions[2])
        self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
    end



    self.dataset_split = {
        self.dataset[{{1, self.ntrain}}],
        self.dataset[{{self.ntrain + 1, self.ntrain + self.nval}}],
        self.dataset[{{self.ntrain + self.nval + 1, self.ntrain + self.nval + self.ntest}}],
    }

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}

    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.dataset_split:size(1), self.dataset_split:size(2), self.dataset_split:size(3)))
    collectgarbage()
    return self
end

function WikiBatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function WikiBatchLoader:next_batch(split_index)
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
    local x = batch[{{}, {1, self.seq_length-1}}]
    local y = batch[{{}, {2, self.seq_length}}]
    return x, y
end

return WikiBatchLoader
