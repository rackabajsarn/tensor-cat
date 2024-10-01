// static/js/script.js

document.addEventListener('DOMContentLoaded', () => {
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
    document.getElementById('save-button').addEventListener('click', () => {
        const filename = document.getElementById('modalImage').getAttribute('data-filename');
        const mode = document.getElementById('modalImage').getAttribute('data-mode');
        saveImage(filename, mode);
    });

    document.getElementById('back-button').addEventListener('click', () => {
        closeModal();
    });

    // Delete Button
    document.getElementById('delete-button').addEventListener('click', () => {
        const filename = document.getElementById('modalImage').getAttribute('data-filename');
        const mode = document.getElementById('modalImage').getAttribute('data-mode');
        deleteImage(filename, mode);
    });

    // Filter Buttons
    const filterButtons = document.querySelectorAll('.filter-button');
    const allButton = document.querySelector('.filter-button[data-filter="all"]');

    filterButtons.forEach(button => {
        button.addEventListener('click', () => {
            if (button === allButton) {
                handleAllButtonClick();
            } else {
                handleFilterButtonClick(button);
            }
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
        }
    })
    .catch(error => {
        console.error('Error fetching labels:', error);
    });
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

// Function to handle "All" button click
function handleAllButtonClick() {
    const filterButtons = document.querySelectorAll('.filter-button');
    const allButton = document.querySelector('.filter-button[data-filter="all"]');

    if (allButton.classList.contains('active')) {
        // "All" is already active; do nothing
        return;
    }

    // Deactivate all other filters
    filterButtons.forEach(btn => {
        if (btn !== allButton) {
            btn.classList.remove('active');
        }
    });

    // Activate "All"
    allButton.classList.add('active');

    // Show all images
    filterImages([]);
}

// Function to handle individual filter button clicks
function handleFilterButtonClick(button) {
    const allButton = document.querySelector('.filter-button[data-filter="all"]');

    // Toggle the clicked filter
    button.classList.toggle('active');

    // If any filter is active, deactivate "All"
    if (button.classList.contains('active')) {
        allButton.classList.remove('active');
    }

    // Collect all active filters
    const activeFilters = Array.from(document.querySelectorAll('.filter-button.active'))
        .filter(btn => btn.getAttribute('data-filter') !== 'all')
        .map(btn => btn.getAttribute('data-filter'));

    if (activeFilters.length === 0) {
        // No filters active; activate "All"
        allButton.classList.add('active');
        filterImages([]);
    } else {
        filterImages(activeFilters);
    }
}

// Function to filter images based on active filters
function filterImages(activeFilters) {
    const imageItems = document.querySelectorAll('.image-item');

    imageItems.forEach(item => {
        if (activeFilters.length === 0) {
            // No filters; show all
            item.classList.remove('hide');
        } else {
            // Show image if it matches any active filter
            const matches = activeFilters.some(filter => {
                return item.getAttribute(`data-${filter}`) === 'yes';
            });
            if (matches) {
                item.classList.remove('hide');
            } else {
                item.classList.add('hide');
            }
        }
    });
}