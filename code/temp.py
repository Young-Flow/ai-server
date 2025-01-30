class CTRDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None, min_threshold=4):
        # 파일에서 열 개수를 확인
        df = pd.read_csv(data_path, sep='\t',header = None)
        self.NUM_FEATS = len(df.columns) - 1  
        print(f"NUM_FEATS 설정됨: {self.NUM_FEATS}")

        self.count, self.data = 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        # 데이터 로드
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        
        # Feature Mapper 초기화
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(feat_values)}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}
        
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims)[:-1]))
    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y'][0]

def load_ctr(batch_size=128):
    train_data = CTRDataset(os.path.join(data_path, 'train.csv'))
    test_data = CTRDataset(os.path.join(data_path, 'test.csv'),
                       feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
    train_iter = torch.utils.data.DataLoader(
        train_data, shuffle=True, drop_last=False, batch_size=batch_size)
    test_iter = torch.utils.data.DataLoader(
        test_data, shuffle=False, drop_last=False, batch_size=batch_size)
    return train_data, test_data, train_iter, test_iter
train_data, test_data, train_iter, test_iter = load_ctr(batch_size=128)