require 'nn'

Constant, parent = torch.class('nn.Constant', 'nn.Module')

function Constant:__init(const)
    self.output = const
end

function Constant:updateOutput(input)
    return self.output
end

function Constant:updateGradInput(input, gradOutput)
    self.gradInput = torch.zeros(input:size())
    return self.gradInput
end
