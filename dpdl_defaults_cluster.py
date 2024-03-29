
MIDOG_DEFAULT_PATH =  r'/home/janus/iwb6-datasets/MIDOG2021/'

JSON_FILE_PATH = r'/home/janus/iwb6-datasets/MIDOG2021/'
MIDOG_JSON_FILE = r'MIDOG.json'


job_id = 787680  #775413 #786183 #785676 #786174
epoch = 299
step = 35700 #35700 #23800 #35700

prob_threshold = 0.3

sample_weight = [0.5, 0.5]

TEST_CHK_POINT_PATH = f'/home/hpc/rzku/mlvl132h/{job_id}/checkpoints/'
TEST_CHK_POINT =  TEST_CHK_POINT_PATH+f'epoch={epoch}-step={step}.ckpt'


CHK_POINT_PATH = './output/checkpoints/'
CHK_POINT = ''


LOGS_PATH = './output/'

# anno_dict = {1: "mitotic figure", 2: "impostor"}

MITOTIC = 1
IMPOSTER =2

# Scanner Source
# tiffs   1- 50:
# tiffs  51-100:
# tiffs 101-150:
# tiffs 151-200:  -- no labels

TRAINING_IDS = [
    "001.tiff",
    "003.tiff",
    "004.tiff",
    "007.tiff",
    "008.tiff",
    "009.tiff",
    "012.tiff",
    "013.tiff",
    "014.tiff",
    "015.tiff",
    "016.tiff",
    "017.tiff",
    "018.tiff",
    "019.tiff",
    "020.tiff",
    "021.tiff",
    "022.tiff",
    "023.tiff",
    "025.tiff",
    "026.tiff",
    "028.tiff",
    "029.tiff",
    "030.tiff",
    "031.tiff",
    "033.tiff",
    "034.tiff",
    "035.tiff",
    "036.tiff",
    "037.tiff",
    "039.tiff",
    "040.tiff",
    "041.tiff",
    "042.tiff",
    "043.tiff",
    "044.tiff",
    "045.tiff",
    "047.tiff",
    "048.tiff",
    "049.tiff",
    "051.tiff",
    "052.tiff",
    "053.tiff",
    "054.tiff",
    "055.tiff",
    "056.tiff",
    "059.tiff",
    "060.tiff",
    "061.tiff",
    "062.tiff",
    "064.tiff",
    "066.tiff",
    "067.tiff",
    "068.tiff",
    "069.tiff",
    "070.tiff",
    "071.tiff",
    "072.tiff",
    "073.tiff",
    "075.tiff",
    "076.tiff",
    "077.tiff",
    "078.tiff",
    "079.tiff",
    "080.tiff",
    "081.tiff",
    "082.tiff",
    "084.tiff",
    "085.tiff",
    "087.tiff",
    "088.tiff",
    "089.tiff",
    "090.tiff",
    "092.tiff",
    "093.tiff",
    "095.tiff",
    "096.tiff",
    "097.tiff",
    "098.tiff",
    "099.tiff",
    "101.tiff",
    "102.tiff",
    "103.tiff",
    "104.tiff",
    "105.tiff",
    "107.tiff",
    "108.tiff",
    "109.tiff",
    "110.tiff",
    "112.tiff",
    "113.tiff",
    "115.tiff",
    "118.tiff",
    "119.tiff",
    "120.tiff",
    "121.tiff",
    "122.tiff",
    "124.tiff",
    "125.tiff",
    "126.tiff",
    "127.tiff",
    "128.tiff",
    "129.tiff",
    "130.tiff",
    "131.tiff",
    "132.tiff",
    "133.tiff",
    "135.tiff",
    "136.tiff",
    "137.tiff",
    "138.tiff",
    "139.tiff",
    "140.tiff",
    "141.tiff",
    "143.tiff",
    "144.tiff",
    "146.tiff",
    "147.tiff",
    "149.tiff",
    "150.tiff",
    "152.tiff",
    "153.tiff",
    "154.tiff",
    "155.tiff",
    "156.tiff",
    "157.tiff",
    "158.tiff",
    "160.tiff",
    "161.tiff",
    "162.tiff",
    "163.tiff",
    "164.tiff",
    "165.tiff",
    "166.tiff",
    "167.tiff",
    "168.tiff",
    "169.tiff",
    "170.tiff",
    "173.tiff",
    "174.tiff",
    "176.tiff",
    "177.tiff",
    "179.tiff",
    "180.tiff",
    "181.tiff",
    "183.tiff",
    "186.tiff",
    "187.tiff",
    "189.tiff",
    "190.tiff",
    "191.tiff",
    "192.tiff",
    "193.tiff",
    "194.tiff",
    "195.tiff",
    "196.tiff",
    "197.tiff",
    "198.tiff",
    "199.tiff",
    "200.tiff",
]
TEST_IDS = [
    "002.tiff",
    "005.tiff",
    "010.tiff",
    "011.tiff",
    "024.tiff",
    "027.tiff",
    "032.tiff",
    "038.tiff",
    "046.tiff",
    "050.tiff",
    "057.tiff",
    "058.tiff",
    "063.tiff",
    "065.tiff",
    "074.tiff",
    "083.tiff",
    "086.tiff",
    "091.tiff",
    "094.tiff",
    "100.tiff",
    "106.tiff",
    "111.tiff",
    "114.tiff",
    "116.tiff",
    "117.tiff",
    "123.tiff",
    "134.tiff",
    "142.tiff",
    "145.tiff",
    "148.tiff",
    "151.tiff",
    "171.tiff",
    "175.tiff",
    "178.tiff",
    "184.tiff",
    "172.tiff",
    "182.tiff",
    "188.tiff",
    "185.tiff",
    "159.tiff",
]

# TEST_IDS = [str(i)+'.tiff' for i in range(101, 151)]
