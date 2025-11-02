/**
 * SmartSelect Platform - Main JavaScript Module
 * Intelligent Feature Optimization System
 * Developed by: Layla Abdallah (UI/UX Designer) & Fadi Younes (API Engineer)
 */

// Global state management
let uploadedDataset = null;
let optimizationResults = null;

// Initialize application on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('SmartSelect Platform Initialized - v2.0');
    initializeApplication();
});

/**
 * Initialize the application environment
 */
function initializeApplication() {
    // Apply entrance animations
    document.querySelectorAll('.main-container').forEach(container => {
        container.classList.add('fade-in');
    });
    
    // Enhanced smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchorElement => {
        anchorElement.addEventListener('click', function (event) {
            event.preventDefault();
            const targetElement = document.querySelector(this.getAttribute('href'));
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add interactive hover effects
    enhanceInteractivity();
}

/**
 * Display loading overlay with custom message
 */
function showLoading(displayMessage = 'Processing optimization...') {
    const overlayElement = document.createElement('div');
    overlayElement.id = 'loadingOverlay';
    overlayElement.className = 'loading-overlay';
    overlayElement.innerHTML = `
        <div class="loading-content">
            <div class="loading-spinner"></div>
            <h4 class="mt-3 fw-bold">${displayMessage}</h4>
            <p class="text-muted">Please wait while we analyze your data</p>
        </div>
    `;
    document.body.appendChild(overlayElement);
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.remove();
    }
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    const alertClass = `alert-${type}-custom`;
    const notification = document.createElement('div');
    notification.className = `alert ${alertClass} alert-dismissible fade show`;
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.zIndex = '10000';
    notification.style.minWidth = '300px';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

/**
 * Format number with commas
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Calculate percentage
 */
function calculatePercentage(value, total) {
    if (total === 0) return 0;
    return ((value / total) * 100).toFixed(2);
}

/**
 * Validate file before upload
 */
function validateFile(file) {
    const validExtensions = ['csv', 'xlsx', 'xls'];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    
    if (!validExtensions.includes(fileExtension)) {
        showNotification('Invalid file type. Please upload CSV or Excel file.', 'danger');
        return false;
    }
    
    // Check file size (max 16MB)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showNotification('File is too large. Maximum size is 16MB.', 'danger');
        return false;
    }
    
    return true;
}

/**
 * Upload file to server
 */
async function uploadFile(file) {
    if (!validateFile(file)) {
        return null;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showLoading('Uploading file...');
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        hideLoading();
        
        if (!response.ok) {
            throw new Error('Upload failed');
        }
        
        const data = await response.json();
        
        if (data.success || data.filename) {
            showNotification('File uploaded successfully!', 'success');
            uploadedFile = data;
            return data;
        } else {
            throw new Error(data.error || 'Upload failed');
        }
        
    } catch (error) {
        hideLoading();
        showNotification(`Upload error: ${error.message}`, 'danger');
        return null;
    }
}

/**
 * Execute evolutionary optimization algorithm
 */
async function runEvolutionaryOptimization(configParams) {
    try {
        showLoading('Executing bio-inspired optimization... This process may require several minutes.');
        
        const apiResponse = await fetch('/api/run-ga', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(configParams)
        });
        
        hideLoading();
        
        if (!apiResponse.ok) {
            throw new Error('Optimization execution failed');
        }
        
        const resultData = await apiResponse.json();
        
        if (resultData.success) {
            showNotification('Evolutionary optimization completed successfully!', 'success');
            optimizationResults = resultData;
            return resultData;
        } else {
            throw new Error(resultData.error || 'Optimization process failed');
        }
        
    } catch (error) {
        hideLoading();
        showNotification(`Optimization Error: ${error.message}`, 'danger');
        return null;
    }
}

/**
 * Run comparison analysis
 */
async function runComparison() {
    try {
        showLoading('Running comparison analysis...');
        
        const response = await fetch('/api/comparison', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        });
        
        hideLoading();
        
        if (!response.ok) {
            throw new Error('Comparison failed');
        }
        
        const data = await response.json();
        
        if (data.success) {
            showNotification('Comparison completed successfully!', 'success');
            return data;
        } else {
            throw new Error(data.error || 'Comparison failed');
        }
        
    } catch (error) {
        hideLoading();
        showNotification(`Comparison Error: ${error.message}`, 'danger');
        return null;
    }
}

/**
 * Export results to CSV
 */
function exportToCSV(data, filename = 'results.csv') {
    const csv = convertToCSV(data);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
    showNotification('Results exported successfully!', 'success');
}

/**
 * Convert data to CSV format
 */
function convertToCSV(data) {
    // Simple CSV conversion
    // This can be enhanced based on data structure
    let csv = '';
    
    if (Array.isArray(data)) {
        // Array of objects
        if (data.length > 0) {
            // Header
            csv += Object.keys(data[0]).join(',') + '\n';
            
            // Rows
            data.forEach(row => {
                csv += Object.values(row).join(',') + '\n';
            });
        }
    } else if (typeof data === 'object') {
        // Single object
        csv += Object.keys(data).join(',') + '\n';
        csv += Object.values(data).join(',') + '\n';
    }
    
    return csv;
}

/**
 * Create chart using Chart.js
 */
function createChart(canvasId, config) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) {
        console.error(`Canvas ${canvasId} not found`);
        return null;
    }
    
    return new Chart(ctx, config);
}

/**
 * Update chart data
 */
function updateChart(chart, newData) {
    if (!chart) return;
    
    chart.data = newData;
    chart.update();
}

/**
 * Animate numbers (count up effect)
 */
function animateNumber(element, targetValue, duration = 1000) {
    const startValue = 0;
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const currentValue = startValue + (targetValue - startValue) * progress;
        element.textContent = Math.floor(currentValue);
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

/**
 * Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Utility: Get query parameter
 */
function getQueryParameter(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

/**
 * Utility: Copy to clipboard
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard!', 'info');
    }).catch(err => {
        console.error('Failed to copy:', err);
    });
}

/**
 * Enhanced interactivity for UI elements
 */
function enhanceInteractivity() {
    // Add ripple effect to buttons
    document.querySelectorAll('.btn').forEach(button => {
        button.addEventListener('click', createRippleEffect);
    });
}

/**
 * Create ripple effect on button click
 */
function createRippleEffect(event) {
    const button = event.currentTarget;
    const ripple = document.createElement('span');
    const rect = button.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = event.clientX - rect.left - size / 2;
    const y = event.clientY - rect.top - size / 2;
    
    ripple.style.width = ripple.style.height = size + 'px';
    ripple.style.left = x + 'px';
    ripple.style.top = y + 'px';
    ripple.classList.add('ripple');
    
    button.appendChild(ripple);
    
    setTimeout(() => ripple.remove(), 600);
}

// Export global API for external scripts
window.SmartSelectAPI = {
    showLoading,
    hideLoading,
    showNotification,
    uploadFile,
    runEvolutionaryOptimization,
    runComparison,
    exportToCSV,
    createChart,
    updateChart,
    animateNumber,
    formatNumber,
    calculatePercentage
};

