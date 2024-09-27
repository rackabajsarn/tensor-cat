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

    // Filter Buttons
    const filterButtons = document.querySelectorAll('.filter-button');
    filterButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove 'active' class from all filter buttons
            filterButtons.forEach(btn => btn.classList.remove('active'));
            // Add 'active' class to the clicked button
            button.classList.add('active');

            const filter = button.getAttribute('data-filter');
            filterImages(filter);
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

// Function to save image
function saveImage(filename, mode) {
    fetch('/update_label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 'filename': filename, 'action': 'save', 'mode': mode })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Image saved successfully.');
            closeModal();
            // Reload the page to reflect changes
            window.location.reload();
        } else {
            alert('Failed to save image: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while saving the image.');
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

// Function to filter images based on selected label
function filterImages(filter) {
    const imageItems = document.querySelectorAll('.image-item');

    imageItems.forEach(item => {
        if (filter === 'all') {
            item.classList.remove('hide');
        } else {
            if (item.getAttribute(`data-${filter}`) === 'yes') {
                item.classList.remove('hide');
            } else {
                item.classList.add('hide');
            }
        }
    });
}

