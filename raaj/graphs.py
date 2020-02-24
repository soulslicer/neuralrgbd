import numpy as np
import matplotlib.pyplot as plt

e1 = {
    "name": "E1",
    "command:": "python train_raaj.py --exp_name raaj --nepoch 50 --RNet --sigma_soft_max 10 --LR 1e-5 --t_win 2 --d_min 1 --d_max 60 --feature_dim 64 --ndepth 64 --grad_clip --grad_clip_max 2. --dataset kitti --dataset_path ../data/datasets/kitti/ --batch_size 4 --qpower 1.5 --ngpu 4",
    "extra": "No other additions used",
    "rmse_1": [6.133664131164551, 5.113693714141846, 3.2601609230041504, 2.790785312652588, 2.5916261672973633, 2.5746121406555176, 2.154703378677368, 2.5126962661743164, 3.825437545776367, 1.8753794431686401, 1.6302672624588013, 2.0107502937316895, 1.746065616607666, 1.75367271900177, 1.6509853601455688, 1.7430684566497803, 2.167996883392334, 2.39646577835083, 2.7322654724121094, 1.7201595306396484, 1.4551465511322021, 1.3139568567276, 1.7090637683868408, 1.6618636846542358, 1.6475632190704346, 1.737297773361206, 1.7481231689453125, 1.873224139213562, 2.186741828918457, 2.435607671737671, 2.228748321533203, 1.4979716539382935, 1.617284893989563, 1.334232211112976, 1.5208946466445923, 2.3780248165130615, 1.4844774007797241, 1.6849530935287476, 1.9922256469726562, 1.8409851789474487, 1.9034672975540161, 1.7460875511169434, 2.05230712890625, 1.9081289768218994, 1.2612731456756592, 1.2799099683761597, 1.459283471107483, 1.755928874015808, 1.6186031103134155, 1.9472126960754395, 1.9789122343063354, 1.5952081680297852, 1.6770752668380737, 1.7153480052947998, 2.0504205226898193, 1.7106472253799438, 1.3430277109146118, 1.3084198236465454, 1.4044947624206543, 2.010033130645752, 1.5985338687896729, 1.6141858100891113, 2.1902098655700684, 2.336437463760376, 1.9946836233139038],
    "sil_1": [0.3214869201183319, 0.21980665624141693, 0.10794206708669662, 0.0927872359752655, 0.08171632885932922, 0.08408308774232864, 0.07111264765262604, 0.08329050242900848, 0.12157554179430008, 0.061197903007268906, 0.05451114848256111, 0.05950487032532692, 0.05016503110527992, 0.04908602684736252, 0.04552662745118141, 0.055791839957237244, 0.06407381594181061, 0.06759937852621078, 0.07748855650424957, 0.05029146000742912, 0.03972082585096359, 0.037259701639413834, 0.04988386482000351, 0.04763754829764366, 0.044987473636865616, 0.04949674755334854, 0.049888789653778076, 0.05239252746105194, 0.06647752225399017, 0.07329265028238297, 0.0638774037361145, 0.04405856877565384, 0.04590828716754913, 0.037508100271224976, 0.0436871312558651, 0.06875793635845184, 0.04101075977087021, 0.04792313650250435, 0.056462083011865616, 0.053868621587753296, 0.0554785318672657, 0.048213791102170944, 0.06108070909976959, 0.05601125210523605, 0.03432323783636093, 0.03649658337235451, 0.04243806004524231, 0.048681244254112244, 0.045633263885974884, 0.05577845871448517, 0.057761404663324356, 0.045673903077840805, 0.04900136590003967, 0.04994284734129906, 0.059958264231681824, 0.049127981066703796, 0.036240190267562866, 0.03671925142407417, 0.03921973332762718, 0.05565105006098747, 0.04662192240357399, 0.04506091773509979, 0.0628632977604866, 0.0657007023692131, 0.05521024763584137],
    "test_interval": 5000,
    "rmse": [5.093856334686279, 4.498722076416016, 3.083822727203369, 2.7413127422332764, 2.699984073638916, 2.597834587097168, 2.502667188644409, 2.40141224861145, 3.5991766452789307, 2.275947332382202, 2.1963088512420654, 2.1019134521484375, 2.2240593433380127, 2.21748948097229, 2.290454149246216, 2.2462806701660156, 2.271103620529175, 2.185169219970703, 2.7858524322509766, 2.1395342350006104, 1.9764498472213745, 2.062957763671875, 2.163479804992676, 2.016000270843506, 1.8703045845031738, 2.0646743774414062, 1.8944143056869507, 2.0399537086486816, 2.13730788230896, 2.215651035308838, 2.3261260986328125, 1.9579334259033203, 2.0737788677215576, 1.9619027376174927, 1.9028606414794922, 2.306797504425049, 1.9389979839324951, 1.914262056350708, 2.009633779525757, 1.9145698547363281, 2.2240750789642334, 2.0640437602996826, 2.088416576385498, 1.900465488433838, 1.8838390111923218, 1.9438542127609253, 1.928377628326416, 2.0189850330352783, 1.970539927482605, 1.930024266242981, 1.9758362770080566, 1.9911922216415405, 1.9003195762634277, 1.9827296733856201, 1.9804960489273071, 1.8695687055587769, 1.864070177078247, 1.9628667831420898, 1.8480559587478638, 2.0715315341949463, 1.9965542554855347, 2.025421142578125, 2.006564140319824, 2.110764265060425, 1.9030420780181885],
    "sil": [0.26158812642097473, 0.1967812180519104, 0.11199184507131577, 0.09861795604228973, 0.09121604263782501, 0.08876412361860275, 0.08955056220293045, 0.0839523896574974, 0.12163680791854858, 0.0854288637638092, 0.08202575892210007, 0.07952307909727097, 0.07677242904901505, 0.07840321958065033, 0.08456078916788101, 0.07965028285980225, 0.07862308621406555, 0.07501637190580368, 0.09091439843177795, 0.0761035606265068, 0.07181143015623093, 0.07430735230445862, 0.075571209192276, 0.0726388469338417, 0.06761518120765686, 0.07377977669239044, 0.06762176752090454, 0.07432842254638672, 0.07548517733812332, 0.07138464599847794, 0.07756990194320679, 0.0711631178855896, 0.07281386852264404, 0.07083477079868317, 0.07010123133659363, 0.07605704665184021, 0.07219196110963821, 0.06702496111392975, 0.07214230298995972, 0.0678308755159378, 0.07449886947870255, 0.06777307391166687, 0.07074231654405594, 0.06976455450057983, 0.06713029742240906, 0.07206238806247711, 0.07098595798015594, 0.07062333822250366, 0.07174233347177505, 0.06837059557437897, 0.06968435645103455, 0.0753200352191925, 0.06763149797916412, 0.06898319721221924, 0.0694018229842186, 0.06712765991687775, 0.06486472487449646, 0.06982579082250595, 0.07017087936401367, 0.0713549330830574, 0.07224582135677338, 0.07089046388864517, 0.06899663805961609, 0.07482430338859558, 0.06527712196111679]
}

e2 = {
    "name": "E2",
    "command:": "python train_raaj.py --exp_name raaj_mf1 --nepoch 60 --RNet --sigma_soft_max 10 --LR 1e-5 --t_win 2 --d_min 1 --d_max 60 --feature_dim 64 --ndepth 96 --grad_clip --grad_clip_max 2. --dataset kitti --dataset_path ../data/datasets/kitti/ --batch_size 4 --qpower 1.5 --ngpu 4 --test --pre_trained_folder outputs/saved_models/raaj_mf1",
    "extra": "No other additions used",
    "test_interval": 5000,
    "rmse": [5.195703983306885, 4.65868616104126, 3.5377135276794434, 2.9341824054718018, 2.706258773803711, 2.7388365268707275, 2.478070020675659, 3.417593002319336, 2.337862730026245, 2.296541213989258, 2.2968318462371826, 2.1265294551849365, 1.9956861734390259, 2.0230345726013184, 2.0877022743225098, 2.2974414825439453, 2.185784339904785, 2.1518218517303467, 2.120391845703125, 2.8898069858551025, 2.143223285675049, 1.982950210571289, 2.0073487758636475, 1.895858645439148, 2.049379587173462, 2.001389980316162, 2.071648597717285, 2.0930521488189697, 1.9212950468063354, 2.0387630462646484, 2.0552120208740234, 2.197216510772705, 1.9162636995315552, 1.8529934883117676, 2.0352776050567627, 1.8198702335357666, 2.133460760116577, 1.9431431293487549, 1.9804474115371704, 1.93356192111969, 2.0305659770965576, 1.8587629795074463, 1.9496488571166992, 2.105231761932373, 1.9596668481826782, 1.9199841022491455, 1.8950260877609253, 1.9052691459655762, 1.959879755973816, 2.010012626647949, 2.1151413917541504, 2.219541072845459, 1.9793587923049927, 1.9591171741485596, 2.0528318881988525, 1.9239814281463623, 1.7886089086532593, 2.04669189453125, 1.9935311079025269, 1.9541095495224, 1.8900656700134277, 1.8139231204986572],
    "sil": [0.27039363980293274, 0.20631657540798187, 0.12628380954265594, 0.10728859156370163, 0.09529860317707062, 0.10000000894069672, 0.08586668968200684, 0.12243708968162537, 0.08655080199241638, 0.08091390877962112, 0.08397180587053299, 0.0800696313381195, 0.07430945336818695, 0.07086760550737381, 0.07383409142494202, 0.08192123472690582, 0.07794768363237381, 0.07369215041399002, 0.07508103549480438, 0.10418067872524261, 0.07889533787965775, 0.07048554718494415, 0.07327713817358017, 0.06834347546100616, 0.0737384706735611, 0.07064533978700638, 0.07042582333087921, 0.0725897029042244, 0.06785178929567337, 0.06831105053424835, 0.06952415406703949, 0.08038594573736191, 0.0697150006890297, 0.06809490919113159, 0.0752694308757782, 0.06555580347776413, 0.07347354292869568, 0.06977535784244537, 0.0682215541601181, 0.06817197054624557, 0.07164578139781952, 0.06492552161216736, 0.06528902798891068, 0.07303399592638016, 0.07138415426015854, 0.06844847649335861, 0.06652173399925232, 0.06806715577840805, 0.06798312067985535, 0.06797600537538528, 0.07175613194704056, 0.07280725240707397, 0.06797964870929718, 0.0669531375169754, 0.06556729972362518, 0.06749900430440903, 0.06688980758190155, 0.06849171966314316, 0.07143059372901917, 0.07240311801433563, 0.06839664280414581, 0.06294700503349304]
}


e = e2
time = np.arange(0,len(e["rmse"]))*e["test_interval"]
#plt.plot(time, e["rmse_1"])
plt.plot(time, e["sil"])
plt.ylabel(e["name"])

e = e1
time = np.arange(0,len(e["rmse"]))*e["test_interval"]
#plt.plot(time, e["rmse_1"])
plt.plot(time, e["sil"])
plt.ylabel(e["name"])

plt.show()