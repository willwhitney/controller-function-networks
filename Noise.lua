require 'nn'

Noise, parent = torch.class('nn.Noise', 'nn.Module')

function Noise:__init(variance)
    parent.__init(self)
    self.variance = variance
    self.active = true
end

function Noise:updateOutput(input)
    if self.active then
        local noise = input:clone()
        if self.variance == 0 then
            noise:fill(0)
        else
            noise:normal(0, self.variance)
        end

        -- print('noise:', noise)
        self.output = input + noise
        -- print('noise input:', input)
        -- print('noise output:', self.output)
    else
        print("Noise inactive!")
        self.output = input
    end
    return self.output
end

function Noise:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput:clone()
    return self.gradInput
end

function Noise:training()
    self.active = true
end

function Noise:evaluate()
    self.active = false
end
