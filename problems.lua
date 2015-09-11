require 'LayerConfig'

problems = {}
function problems.reverse()
    local input = torch.rand(LC.DIM_DATA)
    local target = torch.Tensor(input:size())
    for i = 1, LC.DIM_DATA do
        target[i] = input[LC.DIM_DATA - i + 1]
    end
    return input, target
end


function problems.sort()
    local input = torch.rand(LC.DIM_DATA)
    local target = torch.sort(input)
    return input, target
end


function problems.funkysort()
    local input = torch.zeros(LC.DIM_DATA)
    input[1] = math.random(1, 3)

    if input[1] == 1 then
        input[{{2, LC.DIM_DATA}}]:copy(torch.rand(LC.DIM_DATA - 1) * 1000)
    else
        input[{{2, LC.DIM_DATA}}]:copy(torch.rand(LC.DIM_DATA - 1) * -1000)
    end

    local target = torch.sort(input)
    return input, target
end


function problems.reduce()
    local input = torch.zeros(LC.DIM_DATA)
    local target = torch.zeros(LC.DIM_DATA)

    for i = 1, LC.DIM_DATA do
        input[i] = math.random(1, 10)

        if i == 1 then
            target[i] = input[i]
        else
            target[i] = target[i-1] + input[i]
        end
    end

    return input, target
end
