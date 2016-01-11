require 'nn'

Renormalize, parent = torch.class('nn.Renormalize', 'nn.Module')

function Renormalize:__init()
    print("Renormalize is not a valid module! Do not use!")
    os.exit(1)

    -- parent.__init(self)
    -- self.eps = 1e-100
end
--
-- function Renormalize:updateOutput(input)
--     self.output = input:clone()
--     for i = 1, input:size(1) do
--         self.output[i] = self.output[i] / (input[i]:norm(1) + self.eps)
--     end
--     if self.output[1][1] ~= self.output[1][1] then
--         print('Made a nan set of weights by normalizing.')
--         print('input:', input)
--         os.exit(1)
--     end
--     return self.output
-- end
--
-- function Renormalize:updateGradInput(input, gradOutput)
--     self.gradInput = gradOutput:clone()
--     for i = 1, input:size(1) do
--         self.gradInput[i] = self.gradInput[i] / (input[i]:norm(1) + self.eps)
--     end
--     if self.gradInput[1][1] ~= self.gradInput[1][1] then
--         print('Made a nan set of normalizing gradients.')
--         print('input:', input)
--         print('gradOutput:', gradOutput)
--         os.exit(1)
--     end
--     return self.gradInput
-- end
