# Laporan Proyek Machine Learning - Zefandion Benaya Teja

## Project Overview

Proyek ini bertujuan untuk membangun sistem rekomendasi anime yang efektif, membantu pengguna menemukan anime baru yang mungkin mereka sukai berdasarkan preferensi dan interaksi masa lalu. Industri anime telah berkembang pesat dengan ribuan judul yang tersedia, membuat pengguna kesulitan untuk menemukan konten yang relevan di tengah banyaknya pilihan. Sistem rekomendasi menjadi solusi krusial untuk meningkatkan user experience dan engagement pada platform streaming atau database anime.

Permasalahan utama yang ingin diselesaikan adalah "information overload" bagi pengguna dan "discovery problem" bagi anime. Pengguna seringkali merasa kewalahan dengan banyaknya pilihan dan membutuhkan panduan personal untuk menemukan konten menarik. Di sisi lain, anime yang bagus mungkin tidak ditemukan oleh target audiensnya jika tidak ada mekanisme rekomendasi yang efektif.

Pendekatan untuk mengatasi masalah ini melibatkan penerapan teknik Machine Learning, khususnya sistem rekomendasi. Dengan menganalisis data preferensi pengguna (rating) dan atribut-atribut anime (genre, tipe), kita dapat memprediksi minat pengguna terhadap anime yang belum pernah mereka tonton. Penelitian menunjukkan bahwa sistem rekomendasi yang dipersonalisasi dapat secara signifikan meningkatkan kepuasan pengguna dan retensi platform.

## Business Understanding

### Problem Statements

1. Kelebihan Informasi (Information Overload): Pengguna dihadapkan pada ribuan judul anime, membuat proses pencarian anime yang sesuai dengan selera pribadi menjadi sangat sulit dan memakan waktu. Ini dapat menyebabkan fatigue dan menurunkan kepuasan pengguna.
2. Rendahnya Tingkat Penemuan Konten Baru (Low Content Discovery): Anime-anime baru atau yang kurang populer, meskipun berkualitas tinggi, mungkin tidak mendapatkan eksposur yang cukup dan sulit ditemukan oleh pengguna yang relevan.
3. Pengalaman Pengguna yang Tidak Dipersonalisasi: Tanpa sistem rekomendasi, semua pengguna mendapatkan daftar anime yang sama (misalnya, berdasarkan popularitas), tanpa mempertimbangkan preferensi individu, yang dapat mengurangi relevansi dan nilai platform bagi pengguna.

### Goals

1. Mempermudah Penemuan Anime: Menyediakan rekomendasi anime yang dipersonalisasi agar pengguna dapat dengan mudah menemukan judul-judul yang relevan dengan selera mereka, mengurangi waktu pencarian dan meningkatkan efisiensi.
2. Meningkatkan Eksposur Anime: Membantu anime baru atau yang kurang populer untuk ditemukan oleh audiens yang tepat, mendukung pertumbuhan ekosistem konten anime.
3. Meningkatkan Kepuasan dan Keterlibatan Pengguna: Memberikan pengalaman yang lebih personal dan relevan, yang pada akhirnya akan meningkatkan engagement dan retensi pengguna pada platform atau layanan anime.

    ### Solution statements
    Untuk meraih tujuan-tujuan di atas, proyek ini mengusulkan dua pendekatan utama dalam membangun sistem rekomendasi:
    1. Content-based Filtering: Pendekatan ini akan merekomendasikan anime kepada pengguna berdasarkan kemiripan atribut-atribut anime itu sendiri (seperti genre, tipe). Jika seorang pengguna menyukai anime dengan genre "Action" dan "Fantasy", sistem akan merekomendasikan anime lain yang memiliki genre serupa.
    2. Collaborative Filtering (dengan Matrix Factorization menggunakan TensorFlow): Pendekatan ini akan merekomendasikan anime berdasarkan pola interaksi dan preferensi dari sekelompok besar pengguna. Sistem akan menemukan pengguna dengan selera serupa dan merekomendasikan anime yang disukai oleh pengguna "tetangga" tersebut, atau menemukan anime yang sering ditonton bersama. Implementasi ini akan memanfaatkan kekuatan TensorFlow untuk membangun model Matrix Factorization yang dapat mempelajari representasi embedding dari pengguna dan anime.

## Data Understanding
Bagian ini menjelaskan dataset yang digunakan, struktur, dan karakteristiknya. Dataset yang digunakan berasal dari Anime Recommendations Database dan terbagi menjadi dua file: anime.csv dan rating.csv.

Sumber Data: Dataset ini dapat diunduh dari Kaggle: https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database 

Jumlah data anime yang tersedia adalah 12294 judul unik, sedangkan data rating mencakup 7.813.737 interaksi rating.

Variabel-variabel pada dataset anime.csv:
- anime_id: ID unik untuk setiap anime. (numerik)
- name: Nama lengkap anime. (teks)
- genre: Genre-genre yang relevan dengan anime, dipisahkan koma (misalnya, "Action, Adventure, Fantasy"). (teks)
- type: Tipe rilis anime (misalnya, TV, Movie, OVA, Special, ONA, Music). (kategorikal)
- episodes: Jumlah episode anime. (teks/numerik, perlu penanganan jika ada 'Unknown')
- rating: Rating rata-rata global untuk anime tersebut dari MyAnimeList. (numerik, float)
- members: Jumlah anggota MyAnimeList yang telah menambahkan anime ini ke daftar mereka. (numerik)

Variabel-variabel pada dataset rating.csv:
- user_id: ID unik untuk setiap pengguna. (numerik)
- anime_id: ID unik dari anime yang diberi rating oleh pengguna. (numerik)
- rating: Rating yang diberikan oleh pengguna untuk anime tersebut (1-10). Nilai -1 menunjukkan bahwa pengguna menonton tetapi tidak memberikan rating. (numerik)

Exploratory Data Analysis (EDA) Insights:
- Distribusi Rating Anime (Global): Sebagian besar anime memiliki rating di atas 6, menunjukkan kecenderungan rating yang baik secara keseluruhan.
- Distribusi Tipe Anime: Tipe 'TV' mendominasi dataset, diikuti oleh 'Movie' dan 'OVA', mencerminkan mayoritas konten yang tersedia.
- Top 10 Genre Anime: 'Comedy', 'Action', 'Fantasy', 'Adventure', dan 'Sci-Fi' adalah genre yang paling sering muncul, menunjukkan popularitas genre-genre ini di komunitas anime.
- Distribusi Jumlah Members: Mayoritas anime memiliki jumlah members yang relatif kecil, dengan sedikit anime yang sangat populer (jumlah members tinggi), menunjukkan adanya long tail distribusi popularitas.
- Distribusi Rating User: Terdapat nilai -1 yang perlu ditangani, mengindikasikan interaksi tanpa rating eksplisit. Rating positif (1-10) menunjukkan kecenderungan rating tinggi (6 ke atas) dari pengguna.

## Data Preparation
Tahapan data preparation dilakukan untuk membersihkan, memfilter, dan mentransformasi data mentah menjadi format yang sesuai untuk pemodelan sistem rekomendasi.

Proses Cleaning Data:
1. Penanganan Missing Values pada anime dataset:
- Kolom genre: Missing values diisi dengan "Unknown Genre". Alasan: Genre adalah fitur penting untuk Content-Based Filtering; mengisi dengan kategori "Unknown" mempertahankan baris data yang lain valid.
- Kolom type: Missing values diisi dengan nilai modus (mode). Alasan: Modus adalah kategori yang paling sering muncul, sehingga mengisi dengan modus akan mempertahankan distribusi tipe yang paling umum.
- Kolom rating: Missing values diisi dengan median rating global. Alasan: Median lebih robust terhadap outlier dibandingkan mean, memberikan representasi rating tengah yang lebih baik untuk anime yang belum diberi rating.
2. Penanganan Duplikat Data:
- Tidak ada duplikasi pada dataset anime.
- Satu duplikasi ditemukan dan dihapus pada dataset rating. Alasan: Duplikasi dapat menyebabkan bias dalam model dan evaluasi.
3. Penanganan Rating -1 pada rating dataset:
- Semua baris dengan nilai rating -1 dihapus. Alasan: Nilai -1 tidak merepresentasikan rating eksplisit dan dapat mengganggu perhitungan kemiripan - serta prediksi rating pada model Collaborative Filtering.
4. Penggabungan Dataset: Dataset rating digabungkan dengan dataset anime berdasarkan anime_id. Baris-baris yang kehilangan informasi name, genre, atau type setelah penggabungan (karena anime_id di rating tidak ada di anime) dihapus. Alasan: Memastikan setiap interaksi rating memiliki konteks anime yang lengkap untuk kedua jenis model.

Persiapan Data untuk Content-Based Filtering:
1. Feature Engineering: Kolom genre dan type digabungkan menjadi satu kolom features untuk menangkap keseluruhan konten deskriptif anime.
    ```python
        anime_cbf['features'] = anime_cbf['genre'] + ' ' + anime_cbf['type']
    ```
2. Text Preprocessing: Karakter khusus (misalnya &#039;) dari kolom name dihapus untuk memastikan kualitas teks yang bersih.
    ```python
        anime_cbf['name'] = anime_cbf['name'].apply(lambda x: re.sub(r'&#\d+;', '', x))
    ```
3. Vektorisasi TF-IDF: Menggunakan TfidfVectorizer untuk mengubah fitur teks (features) menjadi representasi numerik. min_df=3 digunakan untuk mengabaikan kata-kata yang terlalu jarang muncul, membantu mengurangi dimensi dan noise.
    ```python
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=3)
        tfidf_matrix = tfidf_vectorizer.fit_transform(anime_cbf['features'])
    ```
Alasan: TF-IDF efektif dalam menangkap pentingnya kata dalam dokumen relatif terhadap korpus, yang esensial untuk mengukur kemiripan konten.

Persiapan Data untuk Collaborative Filtering (TensorFlow/Keras):
1. Filtering Data:
- Hanya pengguna yang memberikan minimal 50 rating dan anime yang menerima minimal 50 rating yang dipertahankan. Alasan: Mengurangi sparsity matriks dan memfokuskan model pada interaksi yang cukup informatif.
2. Encoding ID: user_id dan anime_id diubah menjadi indeks numerik berurutan (0 hingga N-1). Alasan: TensorFlow Embedding layer memerlukan input integer berurutan.
     ```python
        user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        anime_to_idx = {anime_id: idx for idx, anime_id in enumerate(anime_ids)}
        merged_data_cf_tf['user_encoded'] = merged_data_cf_tf['user_id'].map(user_to_idx)
        merged_data_cf_tf['anime_encoded'] = merged_data_cf_tf['anime_id'].map(anime_to_idx)
    ```
3. Normalisasi Rating: Rating dinormalisasi dari skala 1-10 menjadi 0-1. Alasan: Normalisasi rating dapat membantu stabilitas training model Deep Learning dan mempercepat konvergensi.
    ```python
        min_rating = merged_data_cf_tf['rating'].min()
        max_rating = merged_data_cf_tf['rating'].max()
        merged_data_cf_tf['rating'] = (merged_data_cf_tf['rating'] - min_rating) / (max_rating - min_rating)
    ```
4. Split Data: Data dibagi menjadi training set (80%) dan testing set (20%) untuk melatih dan mengevaluasi model.
     ```python
        X = merged_data_cf_tf[['user_encoded', 'anime_encoded']].values
        y = merged_data_cf_tf['rating'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```
Alasan: Memungkinkan evaluasi performa model pada data yang belum pernah dilihat.

## Modeling
Tahapan ini membahas implementasi dua model sistem rekomendasi: Content-Based Filtering dan Collaborative Filtering menggunakan TensorFlow.

Solusi 1: Content-Based Filtering
- Pendekatan: 
Model ini bekerja dengan menghitung kemiripan antar anime berdasarkan fitur konten (genre dan tipe) menggunakan Cosine Similarity pada matriks TF-IDF. Jika pengguna menyukai anime X, sistem akan merekomendasikan anime Y yang memiliki fitur konten yang sangat mirip dengan X.
- Proses Training:
"Training" pada Content-Based Filtering adalah proses pra-komputasi kemiripan antar item. Ini dilakukan sekali dengan menghitung cosine similarity dari matriks TF-IDF yang telah dibuat.
    Cosine Similarity(A,B)= A⋅B / ∥A∥∥B∥
Dimana A dan B adalah vektor fitur TF-IDF dari dua anime.
- Implementasi:
    ```python
        # Hitung Cosine Similarity antar anime berdasarkan TF-IDF matrix
        cosine_sim = cosine_similarity(tfidf_matrix)

        # Fungsi rekomendasi
        def content_based_recommendations(anime_name, num_recommendations=10):
            # ... (kode lengkap ada di notebook)
            pass

        # Contoh output top-10 recommendation untuk 'Fullmetal Alchemist: Brotherhood'
        # Rekomendasi untuk 'Fullmetal Alchemist: Brotherhood':
        #                                  name                                              genre  type  rating  members
        # 10255                    Fullmetal Alchemist  Action, Adventure, Comedy, Drama, Fantasy, Magic, Milit...     TV    8.33   194383
        # 1238                  Hunter x Hunter (2011)             Action, Adventure, Fantasy, Shounen, Super Power     TV    9.12   425828
        # 1686                      Code Geass: Hangyaku no Lelouch             Action, Mecha, Military, School, Sci-Fi, Super P...     TV    8.83   715151
        # ...
    ```
- Kelebihan Content-Based Filtering:
    - Transparansi: Rekomendasi mudah dijelaskan karena didasarkan pada fitur item yang jelas (misalnya, "direkomendasikan karena genre-nya sama").
    - Tidak Ada Cold-Start Pengguna: Model dapat merekomendasikan item kepada pengguna baru tanpa riwayat interaksi, cukup dengan mengetahui profil preferensi awal atau item pertama yang disukai.
    - Mampu merekomendasikan item yang jarang diberi rating: Karena hanya bergantung pada fitur item, bukan popularitas.
- Kekurangan Content-Based Filtering:
    - Keterbatasan dalam Eksplorasi (Serendipity): Model cenderung merekomendasikan item yang sangat mirip dengan yang sudah diketahui pengguna, sehingga kurang mampu merekomendasikan item yang berbeda tetapi mungkin tetap menarik.
    - Kualitas Tergantung Fitur: Kualitas rekomendasi sangat bergantung pada kekayaan dan deskriptivitas fitur item yang tersedia.
    - Cold-Start Item: Item baru tanpa fitur deskriptif yang cukup akan sulit direkomendasikan.
- Hasil rekomendasi (output) model Content-Based Filtering:
    "
    --- Proses Training Content-Based Filtering: Menghitung Cosine Similarity ---
    Bentuk Cosine Similarity Matrix: (12294, 12294)

    Contoh Cosine Similarity untuk 'Kimi no Na wa.':
    name
    Kimi no Na wa.                                           1.000000
    Aura: Maryuuin Kouga Saigo no Tatakai                    0.961543
    Harmonie                                                 0.891655
    Air Movie                                                0.877767
    Wind: A Breath of Heart (TV)                             0.873937
    Wind: A Breath of Heart OVA                              0.867920
    Kokoro ga Sakebitagatterunda.                            0.864523
    Suki ni Naru Sono Shunkan wo.: Kokuhaku Jikkou Iinkai    0.820877
    Clannad Movie                                            0.789141
    Shakugan no Shana                                        0.769155
    Name: Kimi no Na wa., dtype: float64

    --- Demonstrasi Rekomendasi Content-Based Filtering ---
    Rekomendasi untuk 'Fullmetal Alchemist: Brotherhood':
                                                    name  \
    200                              Fullmetal Alchemist   
    1558   Fullmetal Alchemist: The Sacred Star of Milos   
    402        Fullmetal Alchemist: Brotherhood Specials   
    4264                                  Tide-Line Blue   
    ...
    2997    25174  
    2342    72750  
    2852     6903  
    6163    14989  
    "

​Solusi 2: Collaborative Filtering (Matrix Factorization dengan TensorFlow)
- Pendekatan:
Model ini mempelajari embedding laten untuk setiap pengguna dan setiap anime dari data rating. Rating yang diprediksi adalah hasil perkalian dot product antara embedding user dan embedding anime, ditambah dengan bias user dan anime. Model ini mencari pola preferensi tersembunyi antar pengguna dan antar anime.
- Arsitektur Model:
Model RecommenderNet diimplementasikan dengan Keras. Model ini memiliki dua embedding layer (satu untuk user, satu untuk anime) dan dua bias layer (satu untuk user, satu untuk anime). Inputnya adalah ID user yang dienkode dan ID anime yang dienkode. Outputnya adalah rating prediksi yang dinormalisasi, melalui aktivasi sigmoid.
   ```python
        class RecommenderNet(keras.Model):
            def __init__(self, num_users, num_anime, embedding_size, **kwargs):
                super(RecommenderNet, self).__init__(**kwargs)
                self.user_embedding = layers.Embedding(...)
                self.user_bias = layers.Embedding(...)
                self.anime_embedding = layers.Embedding(...)
                self.anime_bias = layers.Embedding(...)

            def call(self, inputs):
                # ... (implementasi dot product dan penambahan bias)
                return tf.nn.sigmoid(x)
    ```
- Proses Training:
Model RecommenderNet dilatih menggunakan data X_train (pasangan user_encoded, anime_encoded) dan y_train (rating yang dinormalisasi). Model dikompilasi dengan MeanSquaredError sebagai fungsi loss dan Adam optimizer, serta dievaluasi menggunakan RootMeanSquaredError.
    ```python
        model = RecommenderNet(num_users, num_anime, embedding_size=50)
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=64,
            epochs=10, # Misalnya 10 epoch
            validation_data=(X_test, y_test),
            verbose=1
        )
    ```
Proses training meminimalkan perbedaan antara rating yang diprediksi dan rating aktual, sehingga embedding user dan anime secara bertahap belajar representasi yang optimal.
- Implementasi Rekomendasi:
Setelah dilatih, model dapat memprediksi rating untuk setiap pasangan user-anime yang belum diberi rating oleh user. Anime dengan prediksi rating tertinggi kemudian direkomendasikan.
    ``` python
        # Fungsi rekomendasi
        def recommend_for_user(user_id, num_recommendations=10):
            # ... (kode lengkap ada di notebook)
            pass

        # Contoh output top-10 recommendation untuk user ID tertentu
        # Rekomendasi untuk User ID: 22
        #      anime_id                                  name                 genre   type  predicted_rating
        # 0       5114      Fullmetal Alchemist: Brotherhood  Action, Adventure, Drama, Fantasy, Magic, Milit...     TV          9.445209
        # 1       9253                           Steins;Gate                                   Sci-Fi, Thriller     TV          9.398014
        # 2      28977                              Gintama°  Action, Comedy, Historical, Parody, Samurai, S...     TV          9.351231
        # ...
    ```
- Kelebihan Collaborative Filtering (Matrix Factorization):
    - Menemukan Pola Tersembunyi: Mampu menangkap interaksi dan preferensi kompleks yang mungkin tidak terlihat dari fitur konten eksplisit.
    - Eksploratif (Serendipity): Dapat merekomendasikan item yang secara konten berbeda tetapi disukai oleh pengguna dengan selera serupa.
    - Skalabilitas dengan Deep Learning: Implementasi dengan TensorFlow memungkinkan model menangani dataset yang sangat besar dan potensi untuk arsitektur yang lebih kompleks.
- Kekurangan Collaborative Filtering:
    - Masalah Cold-Start: Sulit memberikan rekomendasi untuk pengguna baru (belum ada riwayat rating) atau item baru (belum ada yang memberi rating).
    - Sparsity Data: Jika matriks interaksi sangat jarang (sedikit rating dibandingkan jumlah user-item), performa model dapat terpengaruh.
- Hasil rekomendasi (output) dari model Collaborative Filtering:
    "
    --- Demonstrasi Rekomendasi Collaborative Filtering (TensorFlow) ---
    Rekomendasi untuk User ID: 3
    158/158 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step
        anime_id                                name  \
    2153       820                Ginga Eiyuu Densetsu   
    1217     28977                            Gintama°   
    491      11061              Hunter x Hunter (2011)   
    462       4181                Clannad: After Story   
    283      15417            Gintama&#039;: Enchousen   
    73         918                             Gintama   
    443       1575     Code Geass: Hangyaku no Lelouch   
    454       2904  Code Geass: Hangyaku no Lelouch R2   
    729      17074    Monogatari Series: Second Season   
    1320     24701       Mushishi Zoku Shou 2nd Season   

                                                    genre type  predicted_rating  
    2153                     Drama, Military, Sci-Fi, Space  OVA          9.883978  
    1217  Action, Comedy, Historical, Parody, Samurai, S...   TV          9.841286  
    491             Action, Adventure, Shounen, Super Power   TV          9.817928  
    462   Drama, Fantasy, Romance, Slice of Life, Supern...   TV          9.813111  
    283   Action, Comedy, Historical, Parody, Samurai, S...   TV          9.786941  
    73    Action, Comedy, Historical, Parody, Samurai, S...   TV          9.776947  
    443   Action, Mecha, Military, School, Sci-Fi, Super...   TV          9.768796  
    454   Action, Drama, Mecha, Military, Sci-Fi, Super ...   TV          9.758965  
    ...
    1244  Adventure, Drama, Mecha, Military, Romance, Sc...  OVA          8.461169  
    372   Drama, Horror, Mystery, Police, Psychological,...   TV          8.461168  
    98    Drama, Fantasy, Romance, Slice of Life, Supern...   TV          8.436505  
    1559                      Comedy, Mystery, Supernatural   TV          8.411389  
    "
## Evaluation
Evaluasi dilakukan untuk mengukur seberapa baik model rekomendasi bekerja dalam memprediksi rating atau menghasilkan rekomendasi yang relevan.

- Metrik Evaluasi untuk Content-Based Filtering: Precision@K dan Recall@K

Untuk Content-Based Filtering, metrik yang relevan adalah Precision@K dan Recall@K, yang mengukur kualitas rekomendasi dalam konteks daftar Top-N.

- **Precision@K**

    **Formula Metrik:**
    Precision@K adalah proporsi rekomendasi yang relevan di antara K rekomendasi teratas yang diberikan oleh sistem.
    $$Precision@K = \frac{\text{Jumlah rekomendasi relevan dalam top-K}}{\text{K}}$$

    **Bagaimana Metrik Bekerja:**
    Precision@K mengukur seberapa akurat sistem dalam memberikan rekomendasi yang benar-benar relevan dalam daftar rekomendasi teratas. Semakin tinggi nilai Precision@K, semakin sedikit "sampah" (item tidak relevan) dalam daftar rekomendasi.

    Misalnya, jika dari 10 rekomendasi teratas dari 'Kimi no Na wa.', ternyata delapan judul (tidak termasuk judul 'Kimi no Na wa.' itu sendiri) memiliki genre yang sama (jika menggunakan atribut genre) dengan judul yang dicari, maka precision@10 = 8 ÷ 10 = 0.8.

- **Recall@K**

    **Formula Metrik:**
    Recall@K adalah proporsi item relevan yang berhasil direkomendasikan dalam K rekomendasi teratas, dari semua item relevan yang tersedia.
    $$Recall@K = \frac{\text{Jumlah rekomendasi relevan dalam top-K}}{\text{Total item relevan yang tersedia}}$$

    **Bagaimana Metrik Bekerja:**
    Recall@K mengukur kemampuan sistem untuk menemukan *semua* item relevan yang mungkin diminati pengguna. Semakin tinggi nilai Recall@K, semakin banyak item relevan yang dapat ditemukan oleh sistem.

    #### **Asumsi Relevansi untuk Content-Based Filtering:**
    Dalam konteks ini, sebuah rekomendasi dianggap "relevan" jika anime yang direkomendasikan memiliki setidaknya satu genre yang sama dengan genre dari anime yang menjadi input pencarian.

- **Contoh Implementasi Perhitungan Metrik:**
    ```python
        # Contoh evaluasi untuk 'Fullmetal Alchemist: Brotherhood'
        anime_input_cbf_eval = 'Fullmetal Alchemist: Brotherhood'
        k_value_cbf_eval = 10
        recommended_df_cbf_eval = content_based_recommendations(anime_input_cbf_eval, k_value_cbf_eval)
        precision_cbf_val, recall_cbf_val = calculate_precision_recall_at_k_cbf(anime_input_cbf_eval, k_value_cbf_eval, recommended_df_cbf_eval, anime_cbf)
        print(f"Untuk '{anime_input_cbf_eval}' dengan K={k_value_cbf_eval}:")
        print(f"  Precision@{k_value_cbf_eval}: {precision_cbf_val:.4f}")
        print(f"  Recall@{k_value_cbf_eval}: {recall_cbf_val:.4f}")
    ```
    ```python
        # Contoh evaluasi untuk 'Naruto'
        anime_input_cbf_eval_2 = 'Naruto'
        k_value_cbf_eval_2 = 10
        recommended_df_cbf_eval_2 = content_based_recommendations(anime_input_cbf_eval_2, k_value_cbf_eval_2)
        precision_cbf_val_2, recall_cbf_val_2 = calculate_precision_recall_at_k_cbf(anime_input_cbf_eval_2, k_value_cbf_eval_2, recommended_df_cbf_eval_2, anime_cbf)
        print(f"\nUntuk '{anime_input_cbf_eval_2}' dengan K={k_value_cbf_eval_2}:")
        print(f"  Precision@{k_value_cbf_eval_2}: {precision_cbf_val_2:.4f}")
        print(f"  Recall@{k_value_cbf_eval_2}: {recall_cbf_val_2:.4f}")
    ```

- Hasil Evaluasi Precision@K dan Recall@K untuk Content-Based Filtering:
    Contoh Perhitungan untuk 'Fullmetal Alchemist: Brotherhood' dengan K=10:
    
    Precision@10: 0.6000
    Recall@10: 0.0006

    Interpretasi:
    Precision@10 = 0.6000: Dari 10 anime yang direkomendasikan untuk 'Fullmetal Alchemist: Brotherhood', 6 di antaranya memiliki setidaknya satu genre yang sama dengan anime input. Ini menunjukkan bahwa 60% dari rekomendasi teratas dianggap relevan berdasarkan kriteria genre.
    Recall@10 = 0.0006: Nilai recall yang sangat rendah ini mengindikasikan bahwa meskipun model memberikan beberapa rekomendasi relevan di top-10, ada sejumlah besar anime "relevan" lainnya di seluruh dataset yang tidak berhasil masuk ke daftar rekomendasi teratas ini. Hal ini wajar untuk Content-Based Filtering karena "total item relevan" bisa sangat besar (semua anime dengan genre yang sama) dan model hanya menghasilkan daftar rekomendasi yang relatif pendek (Top-K).

    Contoh Perhitungan untuk 'Naruto' dengan K=10:

    Precision@10: 0.9000
    Recall@10: 0.0009

    Interpretasi:

    Precision@10 = 0.9000: Untuk 'Naruto', 9 dari 10 rekomendasi teratas memiliki setidaknya satu genre yang sama, menunjukkan presisi yang sangat tinggi.
    Recall@10 = 0.0009: Mirip dengan contoh sebelumnya, recall tetap rendah karena banyaknya potensi item relevan di seluruh dataset.

- Ringkasan Makna:
Content-Based Filtering menunjukkan presisi yang baik dalam merekomendasikan anime dengan genre yang serupa. Ini berarti rekomendasi yang diberikan kemungkinan besar akan sesuai dengan selera genre pengguna. Namun, kemampuannya untuk menemukan semua anime relevan (Recall) sangat terbatas karena hanya berfokus pada kemiripan fitur dan menghasilkan daftar rekomendasi yang pendek.


- Metrik Evaluasi untuk Collaborative Filtering (Matrix Factorization dengan TensorFlow): Root Mean Squared Error (RMSE)
    Untuk Collaborative Filtering yang berbasis prediksi rating, metrik utama yang digunakan adalah RMSE.

    Metrik Evaluasi: Root Mean Squared Error (RMSE)
    - Formula Metrik:
    RMSE (Root Mean Squared Error) adalah metrik yang umum digunakan untuk mengevaluasi model regresi, termasuk prediksi rating. RMSE mengukur rata-rata magnitudo kesalahan antara nilai prediksi dan nilai aktual.

    Rumus RMSE adalah sebagai berikut:
        RMSE = SQRT( (1/N) * SIGMA( (rui - ^rui)^2 ) )
    Dimana:
    * N adalah jumlah total rating yang diprediksi.
    * rui adalah rating aktual yang diberikan oleh pengguna u untuk item i.
    * ^rui adalah rating yang diprediksi oleh model untuk pengguna u dan item i.
    
    - Bagaimana Metrik Bekerja:
    RMSE memberikan bobot lebih pada kesalahan prediksi yang besar karena mengkuadratkan selisih antara nilai aktual dan prediksi. Semakin kecil nilai RMSE, semakin baik kinerja model dalam memprediksi rating secara akurat.

    - Hasil Evaluasi RMSE untuk Collaborative Filtering:
    Model dilatih selama 10 epochs pada data training dan dievaluasi pada data testing.

    Loss pada data test: 0.0595
    RMSE pada data test (skala normalisasi 0-1): 0.1882
    RMSE pada skala rating asli (1-10): 1.6942 (diperoleh dari 0.1882 * (10 - 1))

    - Grafik RMSE selama Training dan Validasi:
    ![image](https://github.com/user-attachments/assets/1616b885-5109-4b71-bb1e-e7c9f5bd5c47)

    Gambar 1: Plot RMSE selama proses training dan validasi.    

    - Interpretasi Hasil RMSE:
    Nilai RMSE sebesar 1.6942 pada skala rating asli (1-10) menunjukkan bahwa rata-rata kesalahan prediksi rating oleh model adalah sekitar 1.69. Ini adalah hasil yang cukup baik untuk sistem rekomendasi, menunjukkan bahwa model mampu memprediksi preferensi pengguna dengan tingkat akurasi yang reasonable. Perlu diingat bahwa rating asli adalah integer, sedangkan prediksi model adalah float, yang juga menyumbang pada error. Plot RMSE selama training (Gambar 1) menunjukkan bahwa Train RMSE dan Validation RMSE sama-sama menurun di awal pelatihan, dan kemudian Validation RMSE mulai stagnan atau sedikit meningkat setelah beberapa epoch (~epoch 4-5), sementara Train RMSE terus menurun perlahan. Ini menunjukkan model belajar dengan baik tetapi ada indikasi awal dari overfitting yang perlu diperhatikan, meskipun nilai RMSE secara keseluruhan tetap rendah.

- Kesimpulan Metrik:
Kedua pendekatan memiliki kelebihan dan kekurangan masing-masing. Content-Based Filtering unggul dalam transparansi dan penanganan *cold-start* pengguna, serta menunjukkan **presisi yang baik** dalam merekomendasikan anime dengan genre yang serupa. Ini berarti rekomendasi yang diberikan kemungkinan besar akan sesuai dengan selera genre pengguna. Namun, kemampuannya untuk menemukan *semua* anime relevan (Recall) sangat terbatas karena hanya berfokus pada kemiripan fitur dan menghasilkan daftar rekomendasi yang pendek.

Di sisi lain, Collaborative Filtering dengan Matrix Factorization (TensorFlow) menunjukkan kemampuan yang baik dalam memprediksi rating dan menangkap pola preferensi tersembunyi antar pengguna. Nilai RMSE yang rendah untuk Collaborative Filtering menunjukkan model mampu memprediksi rating secara akurat.

Dengan demikian, kedua model memberikan wawasan yang berbeda namun saling melengkapi tentang kinerja sistem rekomendasi. Precision@K memberikan gambaran tentang relevansi daftar rekomendasi Top-N, sedangkan RMSE mengukur akurasi prediksi rating secara keseluruhan. Pilihan antara keduanya atau kombinasi (Hybrid Recommender System) akan tergantung pada prioritas spesifik dari pengalaman pengguna yang diinginkan.


