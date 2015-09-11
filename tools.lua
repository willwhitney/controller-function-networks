require 'vis'

local extra_preloads = {
    "itorch",
    "iterate",
    "xlua",
    "xerror",
    "sys",
    "crypto",
    "_protect_",
    "xprint",
    "image",
    "locals",
    "loop",
    "xrequire",
    "base64",
    "extra_preloads",
    "show_env"
}
for _, preload in ipairs(extra_preloads) do
    _G._preloaded_[preload] = true
end

show_env = function()
    for k, v in pairs(_G) do
        if not _G._preloaded_[k] then
            local namestr = string.sub(tostring(k), 1, 20)
            local valuestr = tostring(v)
            if type(v) == 'userdata' then
                valuestr = vis.simplestr(v)
            end
            valuestr = string.sub(tostring(valuestr), 1, 20)
            -- print("["..namestr.."]\t"..type(v).."\t\t"..valuestr)
            print("["..namestr.."]\t\t"..valuestr)
        end
    end
end
