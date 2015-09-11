require 'nn'
require 'gnuplot'

require 'tools'
require 'vis'

require 'problems'

torch.manualSeed(1)

SCORE_DECAY = 0.1
BATCH_SIZE = 1000

LEARNING_RATE = 0.1


-- lin = vn.layers[1].container.modules[1].modules[1]
-- lin.weight
-- inputTensor = torch.zeros(10)
-- targetTensor = torch.rand(10)

-- print(vn.layers[1].container)

torch.manualSeed(1)

STEPS = 1000000

scores = torch.Tensor(STEPS)
probAvg = 0
for i = 1, STEPS do
    -- input1 = math.random(1, 5)
    -- input2 = math.random(1, 5)
    --
    -- inputTensor = torch.zeros(LC.DIM_DATA)
    -- inputTensor[input1] = 1
    -- inputTensor[5 + input2] = 1
    --
    -- target = input1 + input2
    -- targetTensor = torch.zeros(LC.DIM_DATA)
    -- targetTensor[target] = 1

    -- print("layer 1 gate weights:", vn.layers[1].container.modules[3]:getParameters():sum())



    -- inputTensor = torch.rand(10)
    -- targetTensor = inputTensor:clone()

    local inputTensor, targetTensor = problems.sort()

    output = vn:forward(inputTensor)
    score = vn:backward(inputTensor, targetTensor)
    print("Score "..i..":", score)
    scores[i] = score
    -- local params, gradParams = vn:getParameters()
    -- print(params)
    -- print(gradParams)
    -- print("Sum of gradParams:", gradParams:sum())

    -- print(vn.layers[1].container.modules[1])
    -- local p, gp = vn.layers[1].container.modules[1]:getParameters()
    -- print("layer 1 data gP:", gp:sum())

    if i % BATCH_SIZE == 0 then
        print("---------------------------------------------------------------")
        print("---------------------------------------------------------------")
        print("---------------------------------------------------------------")
        -- print(vn.layerSequence)
        -- print(output[{{90, 100}}])
        -- print(targetTensor[{{90, 100}}])

        gradJs = torch.Tensor(vn.gradJs)
        print("batch average gradJ:", gradJs:mean())
        print("last 20 avg gradJ:", gradJs[{{gradJs:size(1)-19, gradJs:size(1)}}]:mean())

        vn:updateProbabilisticParameters(LEARNING_RATE * 5 * (1 + i / 20000)^-1)
        -- vn:updateProbabilisticParameters(LEARNING_RATE)
        vn:zeroProbabilisticGradParameters()
    end

    if i % 5000 == 0 then
        print(vn.layerSequence)
        print(vis.prettyError(output - targetTensor))
        -- print(output)
        -- print(targetTensor)
    end

    local prob = vn.layerSequence[#vn.layerSequence].prob
    probAvg = .999 * probAvg + .001 * prob

    -- if prob < probAvg / 5 then
    --     print("Unlikely!")
    --     print(vn.layerSequence)
    -- end
    --
    -- if probAvg > 0.999999 then
    --     break
    -- end

    vn:updateDataParameters(LEARNING_RATE / 100 * (1 + i / 20000)^-1)
    vn:zeroDataGradParameters()
end

gnuplot.plot(scores, '-')


--
-- avgScore = 0
-- for i = 1, 100 do
--     input1 = math.random(0, 4)
--     input2 = math.random(0, 4)
--
--     inputTensor = torch.zeros(10)
--     inputTensor[1] = input1
--     inputTensor[2] = input2
--
--     target = input1 + input2
--     targetTensor = torch.Tensor({target})
--     -- targetTensor = torch.zeros(10)
--     -- targetTensor[target + 1] = 1
--
--
--     currentLayer = 1
--     currentInput = inputTensor:clone()
--     while true do
--         layer = layers[currentLayer]
--         currentOutput = layer:forward(currentInput):clone()
--
--         shouldReturn = torch.bernoulli(currentOutput[RETURN_GATE]) == 1
--         currentInput = currentOutput[{{DATA_START, DATA_END}}]:clone()
--
--         shouldJump = torch.bernoulli(currentOutput[JUMP_GATE]) == 1
--         if shouldJump then
--             rawAddress = currentOutput[{{ADDRESS_START, ADDRESS_END}}]
--             addressDistribution = softmax:forward(rawAddress)
--             currentLayer = distributions.cat.rnd(addressDistribution)[1]
--         else
--             currentLayer = currentLayer + 1
--         end
--
--         if shouldReturn or currentLayer > DIM_ADDRESS then
--             break
--         end
--     end
--     resultTensor = currentInput:clone()
--     result = outputLayer:forward(resultTensor)
--
--     score = criterion:forward(result, targetTensor)
--
--
--     snet:zeroGradParameters()
--     snet:backward(sinputTensor, scriterion:backward(snet.output, starget))
--     snet:updateParameters(0.01)
-- end
