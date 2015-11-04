local OneHot, parent = torch.class('OneHot', 'nn.Module')

function OneHot:__init(outputSize)
    parent.__init(self)
    self.outputSize = outputSize
    -- We'll construct one-hot encodings by using the index method to
    -- reshuffle the rows of an identity matrix. To avoid recreating
    -- it every iteration we'll cache it.
    self._eye = torch.eye(outputSize)
end

function OneHot:updateOutput(input)
    -- print(input)
    if input:nDimension() == 1 then
        self.output:resize(input:size(1), self.outputSize):zero()
        if self._eye == nil then self._eye = torch.eye(self.outputSize) end
        self._eye = self._eye:float()
        local longInput = input:long()
        self.output:copy(self._eye:index(1, longInput))
    elseif input:nDimension() == 2 then
        local output = {}
        for i = 1, input:size(1) do
            -- local current_output = torch.zeros(input:size(2), self.outputSize)
            if self._eye == nil then self._eye = torch.eye(self.outputSize) end
            self._eye = self._eye:float()
            local longInput = input[i]:long()
            local current_output = self._eye:index(1, longInput)
            table.insert(output, current_output)
        end
        -- print(output)
        self.output:resize(#output, input:size(2), self.outputSize)
        for i = 1, #output do
            self.output[i]:copy(output[i])
        end
        -- print(self.output)
    else
        error("OneHot can't handle that dimension of input.")
    end
    return self.output
end
