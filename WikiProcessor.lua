
local WikiProcessor = {}
WikiProcessor.__index = WikiProcessor

function WikiProcessor.create(data_dir, vocab_size, batch_size, seq_length)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    local self = {}
    setmetatable(self, WikiProcessor)

    local v = torch.load('vocab.t7')
    self.vocab = v.vocab
    self.ivocab = v.ivocab
    self.vocab_size = vocab_size
    self.batch_size = batch_size
    self.seq_length = seq_length
end

function WikiProcessor:text_to_tensor(in_textfile, out_tensorfile)
    local timer = torch.Timer()

    -- construct a tensor with all the data
    print('putting data into tensor...')
    local data = {} -- store it into 1D first, then rearrange
    f = io.open(in_textfile, "r")

    local i = 1
    line = f:read("*line")
    while line do
        if line:sub(1, 1) ~= '<' then
            print(i)
            for word in line:gmatch("%w+") do
                table.insert(data, word)
            end
        end
        line = f:read("*line")
        i = i + 1
    end
    f:close()

    function next_valid_sequence(starting_index)
        local i = starting_index
        while true do
            local sequence = torch.Tensor(self.seq_length)
            local is_valid = true
            for sequence_index = 1, self.seq_length do
                if i > #data then
                    return nil
                end

                local word = data[i]
                word = word:lower()
                local encoding = self.vocab[word]

                i = i+1
                if not encoding or encoding > self.vocab_size then
                    is_valid = false
                    break
                else
                    sequence[sequence_index] = encoding
                end

            end
            if is_valid then
                return sequence, i
            end
        end
    end

    function next_batch(starting_index)
        local i = starting_index
        local batch = torch.Tensor(self.batch_size, self.seq_length)
        for batch_index = 1, self.batch_size do
            sequence, i = next_valid_sequence(i)
            if sequence == nil then
                return nil
            end

            batch[batch_index] = sequence
        end

        collectgarbage()
        return batch, i
    end

    local data_table = {}
    local batch_number = 1
    local i = 1

    batch, i = next_batch(i)
    while batch ~= nil do
        print(string.format("%.2f%% (%s/%s)", 100 * i / #data, i, #data))
        table.insert(data_table, batch)
        batch, i = next_batch(i)
    end

    print('Number of batches:', #data_table)

    data_tensor = torch.Tensor(#data_table, self.batch_size, self.seq_length)
    for i, batch in ipairs(data_table) do
        data_tensor[i]:copy(batch)
    end


    -- save output preprocessed file
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, data_tensor)
end

return WikiProcessor
