local ExpectationCriterion, parent = torch.class('nn.ExpectationCriterion', 'nn.Criterion')

function ExpectationCriterion:__init(criterion)
    parent.__init(self)
    self.criterion = criterion
end

function ExpectationCriterion:updateOutput(input, target)
    local probs, outputs = table.unpack(input)
    -- print("probs\n", probs)
    -- print("outputs\n", outputs)
    local loss = 0
    for i = 1, outputs:size(1) do
        -- print(probs[1][i] * self.criterion:forward(outputs[i], target))
        loss = loss + probs[1][i] * self.criterion:forward(outputs[i], target)
    end
    self.output = loss
    return self.output
end

function ExpectationCriterion:updateGradInput(input, target)
    local probs, outputs = table.unpack(input)
    -- print("probs\n", probs)
    -- print("outputs\n", outputs)

    local grad_probs = torch.Tensor(probs:size())
    local grad_outputs = torch.Tensor(outputs:size())

    for i = 1, outputs:size(1) do
        -- have to run criterion forward for correctness,
        -- so caching this doesn't help
        grad_probs[1][i] = self.criterion:forward(outputs[i], target)
        grad_outputs[i] = self.criterion:backward(outputs[i], target) * probs[1][i]
    end
    self.gradInput = {grad_probs, grad_outputs}
    return self.gradInput
end

function ExpectationCriterion:type(type, tensorCache)
    self.criterion:type(type)
    self.losses = nil
    return parent.type(self, type, tensorCache)
end
