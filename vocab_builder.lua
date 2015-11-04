wc_file = io.open('count_1w.txt')

vocab = {}
ivocab = {}

line = wc_file:read("*line")
i = 1
while line ~= nil do -- and i <= 50000 do
    for word in line:gmatch("%w+") do
        vocab[word] = i
        ivocab[i] = word
        print(word, other)
        break
    end
    i = i+1
    line = wc_file:read("*line")
end

vocabfile = {
    vocab = vocab,
    ivocab = ivocab,
}
torch.save('vocab.t7', vocabfile)
