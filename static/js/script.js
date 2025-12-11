// static/js/script.js

// Track current image index for navigation
let currentImageIndex = 0;
let imageList = [];
let currentInferenceInfo = null;

document.addEventListener('DOMContentLoaded', () => {
    // Build the image list from visible thumbnails
    buildImageList();

    // Classification Modal Buttons
    const labelButtons = document.querySelectorAll('.label-button');
    labelButtons.forEach(button => {
        button.addEventListener('click', () => {
            const label = button.getAttribute('data-label');
            const filename = document.getElementById('modalImage').getAttribute('data-filename');
            const mode = document.getElementById('modalImage').getAttribute('data-mode');
            toggleLabel(button, label, filename, mode);
        });
    });

    // Save and Back Buttons
    const saveButton = document.getElementById('save-button');
    if (saveButton) {
        saveButton.addEventListener('click', () => {
            const modalImage = document.getElementById('modalImage');
            if (!modalImage) {
                return;
            }
            const filename = modalImage.getAttribute('data-filename');
            const mode = modalImage.getAttribute('data-mode');
            saveImage(filename, mode);
        });
    }

    const backButton = document.getElementById('back-button');
    if (backButton) {
        backButton.addEventListener('click', () => {
            closeModal();
        });
    }

    const deleteButton = document.getElementById('delete-button');
    if (deleteButton) {
        deleteButton.addEventListener('click', () => {
            const modalImage = document.getElementById('modalImage');
            if (!modalImage) {
                return;
            }
            const filename = modalImage.getAttribute('data-filename');
            const mode = modalImage.getAttribute('data-mode');
            deleteImage(filename, mode);
        });
    }

    // Navigation buttons
    const prevButton = document.getElementById('nav-prev');
    const nextButton = document.getElementById('nav-next');
    
    if (prevButton) {
        prevButton.addEventListener('click', (e) => {
            e.stopPropagation();
            navigateToPrev();
        });
    }
    
    if (nextButton) {
        nextButton.addEventListener('click', (e) => {
            e.stopPropagation();
            navigateToNext();
        });
    }

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        const modal = document.getElementById('classifyModal');
        if (!modal || modal.style.display !== 'block') {
            return;
        }
        
        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            navigateToPrev();
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            navigateToNext();
        } else if (e.key === 'Escape') {
            closeModal();
        }
    });

    // Filter Buttons
    const filterButtons = document.querySelectorAll('.filter-button');

    filterButtons.forEach(button => {
        button.addEventListener('click', () => {
            const filter = button.getAttribute('data-filter');

            filterButtons.forEach(btn => {
                //if (btn.getAttribute('data-filter') !== 'all') {
                    btn.classList.remove('active');
                //}
            });
            button.classList.toggle('active');

            // if (filter === 'all') {
            //     // If 'All' is clicked
            //     // Remove 'active' class from other buttons
            //     filterButtons.forEach(btn => {
            //         if (btn.getAttribute('data-filter') !== 'all') {
            //             btn.classList.remove('active');
            //         }
            //     });
            //     // Activate 'All' button
            //     button.classList.add('active');
            // } else {
            //     // Toggle 'active' on the clicked button
            //     button.classList.toggle('active');
            //     // Remove 'active' from 'All' button
            //     const allButton = document.querySelector('.filter-button[data-filter="all"]');
            //     allButton.classList.remove('active');
            // }

            // If no filters are active, activate 'All' button
            const activeFilters = Array.from(filterButtons)
                .filter(btn => btn.classList.contains('active') && btn.getAttribute('data-filter') !== 'all');

            if (activeFilters.length === 0) {
                const allButton = document.querySelector('.filter-button[data-filter="all"]');
                allButton.classList.add('active');
            }

            // Call the filterImages function
            filterImages();
        });
    });
    
});

// Function to toggle label
function toggleLabel(button, label, filename, mode) {
    fetch('/update_label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 'filename': filename, 'label': label, 'action': 'toggle', 'mode': mode })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const newState = data.labels[label];
            if (newState) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }
        } else {
            alert('Failed to update label: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while updating the label.');
    });
}

// Function to save image without popup
function saveImage(filename, mode) {
    fetch('/update_label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 'filename': filename, 'action': 'save', 'mode': mode })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Optionally, you can update the UI here (e.g., move the image to gallery)
            closeModal();
            // Refresh the classify view or remove the image from the classify list
            window.location.reload(); // Alternatively, implement a more efficient UI update
        } else {
            alert('Failed to save image: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while saving the image.');
    });
}

// Function to delete image
function deleteImage(filename, mode) {
    if (!confirm('Are you sure you want to delete this image?')) {
        return; // User cancelled deletion
    }

    fetch('/delete_image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 'filename': filename, 'mode': mode })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Close the modal
            closeModal();

            // Remove the image from the DOM
            const imageElement = document.querySelector(`.image-item img[src="/image/${mode}/${filename}"]`);
            if (imageElement) {
                imageElement.parentElement.remove();
            }

            // Optionally, display a success message without a popup
            // For example, update a status div or use a toast notification library
            console.log('Image deleted successfully.');
        } else {
            alert('Failed to delete image: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while deleting the image.');
    });
}

// Open Modal Function
function openModal(filename, mode) {
    var modal = document.getElementById("classifyModal");
    var modalImg = document.getElementById("modalImage");
    modal.style.display = "block";
    modalImg.src = `/image/${mode}/${filename}`;
    modalImg.setAttribute('data-filename', filename);
    modalImg.setAttribute('data-mode', mode);

    currentInferenceInfo = null;

    // Update current index for navigation
    buildImageList();
    currentImageIndex = imageList.findIndex(img => img.filename === filename);
    updateNavButtons();

    // Fetch current labels to set button states
    fetch('/update_label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 'filename': filename, 'action': 'get_labels', 'mode': mode })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success && data.labels) {
            const labels = data.labels;
            // Update button states
            document.getElementById('btn-cat').classList.toggle('active', labels['cat']);
            document.getElementById('btn-morris').classList.toggle('active', labels['morris']);
            document.getElementById('btn-entering').classList.toggle('active', labels['entering']);
            document.getElementById('btn-prey').classList.toggle('active', labels['prey']);
            currentInferenceInfo = data.inference || null;
            updateNavCounterDisplay();
        }
    })
    .catch(error => {
        console.error('Error fetching labels:', error);
    });
}

// Build list of visible images for navigation
function buildImageList() {
    imageList = [];
    const items = document.querySelectorAll('.image-item:not(.hide)');
    items.forEach(item => {
        const img = item.querySelector('img');
        if (img) {
            const src = img.getAttribute('src');
            const onclick = img.getAttribute('onclick');
            // Extract filename and mode from onclick="openModal('filename', 'mode')"
            const match = onclick && onclick.match(/openModal\('([^']+)',\s*'([^']+)'\)/);
            if (match) {
                imageList.push({ filename: match[1], mode: match[2] });
            }
        }
    });
}

// Navigate to previous image
function navigateToPrev() {
    if (imageList.length === 0) return;
    currentImageIndex = (currentImageIndex - 1 + imageList.length) % imageList.length;
    const img = imageList[currentImageIndex];
    openModal(img.filename, img.mode);
}

// Navigate to next image
function navigateToNext() {
    if (imageList.length === 0) return;
    currentImageIndex = (currentImageIndex + 1) % imageList.length;
    const img = imageList[currentImageIndex];
    openModal(img.filename, img.mode);
}

// Update navigation button visibility
function updateNavButtons() {
    const prevBtn = document.getElementById('nav-prev');
    const nextBtn = document.getElementById('nav-next');
    
    if (prevBtn) prevBtn.style.display = imageList.length > 1 ? 'flex' : 'none';
    if (nextBtn) nextBtn.style.display = imageList.length > 1 ? 'flex' : 'none';
    updateNavCounterDisplay();
}

function formatConfidence(conf) {
    if (conf === null || conf === undefined) {
        return '';
    }
    const num = Number(conf);
    if (Number.isFinite(num)) {
        return ` (${(num * 100).toFixed(0)}%)`;
    }
    return '';
}

function updateNavCounterDisplay() {
    const counter = document.getElementById('nav-counter');
    if (!counter) return;
    const total = imageList.length;
    const indexText = total > 0 ? `${currentImageIndex + 1} / ${total}` : '--';

    const espLabel = currentInferenceInfo && currentInferenceInfo.esp32_inference
        ? `${currentInferenceInfo.esp32_inference}${formatConfidence(currentInferenceInfo.esp32_confidence)}`
        : ': --';
    const serverLabel = currentInferenceInfo && currentInferenceInfo.server_inference
        ? `${currentInferenceInfo.server_inference}${formatConfidence(currentInferenceInfo.server_confidence)}`
        : ': --';

    counter.innerHTML = `
        <span class="nav-pill"><span class="pill-icon">🛰️</span> ${espLabel}</span>
        <span class="nav-index">${indexText}</span>
        <span class="nav-pill"><span class="pill-icon">🖥️</span> ${serverLabel}</span>
    `;
}

// Close Modal Function
function closeModal() {
    var modal = document.getElementById("classifyModal");
    modal.style.display = "none";
}

// Close modal when clicking outside of modal content
window.onclick = function(event) {
    var modal = document.getElementById("classifyModal");
    if (event.target == modal) {
        modal.style.display = "none";
    }
}

// Function to filter images based on selected labels
function filterImages() {
    const imageItems = document.querySelectorAll('.image-item');
    const filteredImages = document.getElementById('filtered-images');
    const activeFilters = Array.from(document.querySelectorAll('.filter-button.active'))
        .map(btn => btn.getAttribute('data-filter'))
        .filter(filter => filter !== 'all');

    if (activeFilters.length === 0) {
        // No filters active, show all images
        imageItems.forEach(item => {
            item.classList.remove('hide');
        });
    } else {
        imageItems.forEach(item => {
            // Check if the item matches all active filters
            const matchesAll = activeFilters.every(filter => item.getAttribute(`data-${filter}`) === 'yes');

            if (matchesAll) {
                item.classList.remove('hide');
            } else {
                item.classList.add('hide');
            }
        });
    }
    const visibleCount = document.querySelectorAll('.image-item:not(.hide)').length;
    filteredImages.textContent = visibleCount + " images";
}

document.addEventListener('DOMContentLoaded', function () {
    const themeToggle = document.getElementById('theme-toggle');
    const navBar = document.querySelector('.nav-bar');

    const applyTheme = (theme) => {
        const isDark = theme === 'dark';
        document.body.classList.toggle('dark-theme', isDark);
        if (navBar) {
            navBar.classList.toggle('dark-theme', isDark);
        }
    };

    let savedTheme = localStorage.getItem('theme');
    if (!savedTheme) {
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        savedTheme = prefersDark ? 'dark' : 'light';
    }
    applyTheme(savedTheme);

    if (!themeToggle) {
        return;
    }

    themeToggle.checked = savedTheme === 'dark';
    themeToggle.addEventListener('change', function () {
        if (this.checked) {
            applyTheme('dark');
            localStorage.setItem('theme', 'dark');
        } else {
            applyTheme('light');
            localStorage.setItem('theme', 'light');
        }
    });
});