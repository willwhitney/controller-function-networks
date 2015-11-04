f = io.open('data/wiki/wiki_00', 'r')
outfile = io.open('data/wiki/input.txt', 'w')

line = f:read('*line')
while line do
    if line:sub(1, 1) ~= '<' then
        outfile:write(line)
        outfile:write('\n')
    end

    line = f:read('*line')
end
