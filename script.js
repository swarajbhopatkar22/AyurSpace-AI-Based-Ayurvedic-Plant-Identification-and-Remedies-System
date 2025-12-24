document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const result = document.getElementById('result');
    const heatmapBtn = document.getElementById('heatmapBtn');
    const printReportBtn = document.getElementById('printReportBtn');

    // Q&A elements
    const qaInput = document.getElementById('qaInput');
    const qaBtn = document.getElementById('qaBtn');
    const qaAnswer = document.getElementById('qaAnswer');

    let currentLang = 'en';
    let lastUploadedFile = null;

    // Language toggle buttons
    const btnEn = document.getElementById('langEn');
    const btnHi = document.getElementById('langHi');
    const btnMr = document.getElementById('langMr');

    function setLang(lang) {
        currentLang = lang;
        [btnEn, btnHi, btnMr].forEach(b => b.classList.remove('active'));
        if (lang === 'en') btnEn.classList.add('active');
        if (lang === 'hi') btnHi.classList.add('active');
        if (lang === 'mr') btnMr.classList.add('active');
    }

    btnEn.addEventListener('click', () => setLang('en'));
    btnHi.addEventListener('click', () => setLang('hi'));
    btnMr.addEventListener('click', () => setLang('mr'));

    setLang('en');

    // Text dictionary
    function getText(key) {
        const texts = {
            title: { en: 'Medicinal Uses', hi: 'औषधीय उपयोग', mr: 'औषधी उपयोग' },
            remedy: { en: 'Remedy', hi: 'उपचार', mr: 'उपचार' },
            precautions: { en: 'Precautions', hi: 'सावधानियाँ', mr: 'सूचना' }
        };
        return (texts[key] && texts[key][currentLang]) || texts[key]['en'];
    }

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Click to browse
    uploadArea.addEventListener('click', (e) => {
        preventDefaults(e);
        imageInput.click();
    });

    // Drag & drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    uploadArea.addEventListener('dragover', () => {
        uploadArea.style.background = '#d5f4e6';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.background = 'white';
    });

    uploadArea.addEventListener('drop', (e) => {
        uploadArea.style.background = 'white';
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleImageUpload(files[0]);
        }
    });

    // SINGLE imageInput change handler - FILE SAVE + PREDICT
    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            lastUploadedFile = e.target.files[0];
            handleImageUpload(lastUploadedFile);
        }
    });

    function handleImageUpload(file) {
        lastUploadedFile = file;  // store globally
    const formData = new FormData();
    formData.append('file', file);
    formData.append('lang', currentLang);

         const heatmapImg = document.getElementById('heatmapImg');
         const heatmapContainer = document.getElementById('heatmapContainer');

         if (heatmapImg) {
             heatmapImg.src = '';              // purani image hatao
    }
         if (heatmapContainer) {
             heatmapContainer.style.display = 'none';   // container hide karo
    }

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }

             qaInput.value = '';
             qaAnswer.innerHTML = '';

            document.getElementById('plantName').textContent = data.plant_name;
            document.getElementById('confidence').textContent =
                `Accuracy: ${data.confidence.toFixed(2)}%`;

            if (data.low_confidence) {
                alert('Model is not very confident on this image. Try a clearer leaf photo for better results.');
            }

            const plantImg = document.getElementById('plantImage');
            plantImg.innerHTML = `<img src="${URL.createObjectURL(file)}" alt="${data.plant_key}">`;

            // Remedies + keywords
            document.getElementById('remedies').innerHTML = `
                <h3>${getText('title')}:</h3>
                <p><strong>${data.uses.join(', ')}</strong></p>
                <h3>${getText('remedy')}:</h3>
                <p>${data.remedy}</p>
                <h3>${getText('precautions')}:</h3>
                <p><em>${data.precautions}</em></p>
                ${data.keywords && data.keywords.length ? `
                  <h3>Keywords:</h3>
                  <div class="tag-row">
                    ${data.keywords.map(k => `<span class="tag">${k}</span>`).join('')}
                  </div>
                ` : ''}
            `;

            result.classList.remove('hidden');
            heatmapBtn.disabled = false;
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Prediction failed. Please try again.');
            heatmapBtn.disabled = true;
        });
    }

    // Download / Print report
    printReportBtn.addEventListener('click', function() {
        if (result.classList.contains('hidden')) {
            alert('Please get a prediction first!');
            return;
        }
        window.print();
    });

    // HEATMAP BUTTON
    heatmapBtn.addEventListener('click', async function() {
        if (!lastUploadedFile) {
            alert('Please upload an image first!');
            return;
        }

        const btn = this;
        btn.disabled = true;
        btn.innerHTML = '🔄 Generating Heatmap...';
        btn.style.opacity = '0.7';

        const formData = new FormData();
        formData.append('file', lastUploadedFile);

        try {
            const response = await fetch('/gradcam', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.heatmap_image) {
                const ts = Date.now();
                document.getElementById('heatmapImg').src =
                    `/uploads/${data.heatmap_image}?t=${ts}`;
                document.getElementById('heatmapContainer').style.display = 'block';
                document.getElementById('heatmapImg').scrollIntoView({ behavior: 'smooth' });
            } else {
                alert(' Heatmap generation failed. Try another image.');
            }
        } catch (error) {
            console.error('Heatmap error:', error);
            alert(' Network error. Please try again.');
        } finally {
            btn.disabled = false;
            btn.innerHTML = '🌡️ View Heatmap (Model Focus)';
            btn.style.opacity = '1';
        }
    });

    // Q&A button
    qaBtn.addEventListener('click', async () => {
        const q = qaInput.value.trim();
        if (!q) return;

        qaBtn.disabled = true;
        qaBtn.textContent = 'Thinking...';

        try {
            const res = await fetch('/qa', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: q, lang: currentLang })
            });
            const data = await res.json();
            qaAnswer.innerHTML = `
               <p style="margin-bottom: 6px;">
                 <span style="
                   display:inline-flex;align-items:center;
                   background:#e8f5e9;color:#2e7d32;
                   padding:2px 10px;border-radius:999px;
                   font-size:12px;font-weight:600;">
                   Answer
                 </span>
              </p>
              <p style="margin-top:4px;">
            ${data.answer.replace(/\n/g, '<br>')}
  </p>
`;

        } catch (err) {
            console.error(err);
            qaAnswer.textContent = 'Error while answering the question.';
        } finally {
            qaBtn.disabled = false;
            qaBtn.textContent = 'Ask';
        }
    });
});
