require 'lfs'
WikiProcessor = require 'WikiProcessor'

wiki_data_dir = '/Users/will/Dropbox (MIT Startup Club)/datasets/wiki-extracted/AA/'

for _, vocab_size in ipairs{10000, 20000, 30000, 40000, 50000} do
    batchLoader = WikiProcessor.create(
            wiki_data_dir,
            vocab_size,
            30,
            20,
            {0.9, 0.05, 0.05}
        )
    batchLoader:text_to_tensor(wiki_data_dir .. 'output.txt', 'dataset_'..vocab_size..'.t7')
    collectgarbage()
end
