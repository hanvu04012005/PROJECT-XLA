function handleFiles(files) {
  const fileStatus = document.getElementById("file-status");
  const imageGrid = document.getElementById("image-grid");
  const imagePreviewContainer = document.getElementById("image-preview-container");
  const uploadContent = document.querySelector(".upload-content");
  
  imageGrid.innerHTML = "";

  if (!files.length) {
    fileStatus.textContent = "Chưa có ảnh nào được chọn";
    imagePreviewContainer.classList.remove("show");
    uploadContent.style.display = "block";
    return;
  }

  // Validation file types
  const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
  for (let file of files) {
    if (!allowedTypes.includes(file.type)) {
      showError(`File ${file.name} không được hỗ trợ. Chỉ chấp nhận: JPG, PNG, GIF, BMP`);
      return;
    }
  }

  // Ẩn hướng dẫn và hiển thị ảnh
  uploadContent.style.display = "none";
  fileStatus.textContent = `${files.length} ảnh đã chọn`;
  imagePreviewContainer.classList.add("show");

  Array.from(files).forEach((file, index) => {
    const reader = new FileReader();
    reader.onload = e => {
      const img = document.createElement("img");
      img.src = e.target.result;
      img.classList.add("preview-img");
      img.title = file.name;
      imageGrid.appendChild(img);
    };
    reader.readAsDataURL(file);
  });
}

function handleDrop(event) {
  event.preventDefault();
  const uploadArea = document.getElementById("upload-area");
  uploadArea.classList.remove("dragover");
  
  const files = event.dataTransfer.files;
  document.getElementById("file-input").files = files;
  handleFiles(files);
}

function handleDragOver(event) {
  event.preventDefault();
  const uploadArea = document.getElementById("upload-area");
  uploadArea.classList.add("dragover");
}

function handleDragLeave(event) {
  event.preventDefault();
  const uploadArea = document.getElementById("upload-area");
  uploadArea.classList.remove("dragover");
}

function handleObjectChange() {
  const value = document.getElementById("object-select").value;
  const objectNameInput = document.getElementById("object-name");
  const infoOptions = document.getElementById("info-options");
  
  objectNameInput.classList.toggle("hidden", value !== "doi-tuong");
  infoOptions.classList.toggle("hidden", value !== "doi-tuong");
  
  // Clear info when switching modes
  if (value !== "doi-tuong") {
    clearInfo();
  }
}

function handleFunctionChange() {
  const value = document.getElementById("function-select").value;
  const subFunctionSelect = document.getElementById("sub-function-select");
  subFunctionSelect.classList.toggle("hidden", value !== "sac-bien");
}

function showInfo(type) {
  // Có thể xử lý nếu muốn cập nhật dữ liệu trước khi gửi
}

function clearInfo() {
  const infoBox = document.getElementById("info-box");
  infoBox.innerHTML = `
    <div class="no-info">
      <div class="no-info-icon"></div>
      <p>Chưa có thông tin</p>
      <small>Chọn "Đối tượng cụ thể" và loại thông tin để xem</small>
    </div>
  `;
}

function showLoading() {
  const button = document.querySelector('.process-btn');
  button.classList.add('loading');
  button.disabled = true;
}

function hideLoading() {
  const button = document.querySelector('.process-btn');
  button.classList.remove('loading');
  button.disabled = false;
}

function showError(message) {
  const errorDiv = document.createElement('div');
  errorDiv.className = 'error-message';
  errorDiv.textContent = message;
  errorDiv.style.cssText = `
    background: linear-gradient(45deg, #e74c3c, #c0392b);
    color: white;
    padding: 15px 20px;
    border-radius: 12px;
    margin: 15px 0;
    text-align: center;
    font-weight: 500;
    box-shadow: 0 5px 15px rgba(231, 76, 60, 0.3);
    animation: slideIn 0.3s ease;
  `;
  
  const container = document.querySelector('.control-section');
  container.parentNode.insertBefore(errorDiv, container.nextSibling);
  
  setTimeout(() => {
    errorDiv.style.animation = 'slideOut 0.3s ease';
    setTimeout(() => errorDiv.remove(), 300);
  }, 5000);
}

function handleProcess() {
  const files = document.getElementById("file-input").files;
  if (!files.length) {
    showError("Vui lòng chọn ít nhất một ảnh");
    return;
  }

  showLoading();

  const formData = new FormData();
  Array.from(files).forEach(f => formData.append("images", f));
  formData.append("object_type", document.getElementById("object-select").value);
  formData.append("object_name", document.getElementById("object-name").value);
  formData.append("function_mode", document.getElementById("function-select").value);
  formData.append("sub_mode", document.getElementById("sub-function-select").value);
  const infoRadio = document.querySelector("input[name='info']:checked");
  if (infoRadio) formData.append("info_type", infoRadio.value);

  fetch("/process", {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      hideLoading();
      
      if (data.error) {
        showError(data.error);
        return;
      }

      // Hiển thị ảnh kết quả
      const resultArea = document.getElementById("result-image-area");
      if (data.images && data.images.length > 0) {
        resultArea.innerHTML = "";
        resultArea.classList.add("has-content");
        
        data.images.forEach(src => {
          const img = document.createElement("img");
          img.src = src;
          img.classList.add("result-img");
          resultArea.appendChild(img);
        });
      } else {
        resultArea.innerHTML = `
          <div class="no-result">
            <div class="no-result-icon">📷</div>
            <p>Không có ảnh kết quả</p>
            <small>Vui lòng thử lại với ảnh khác</small>
          </div>
        `;
        resultArea.classList.remove("has-content");
      }

      // Cập nhật bảng phân loại
      const table = document.getElementById("classification-table");
      if (data.counts && Object.keys(data.counts).length > 0) {
        table.innerHTML = "";
        for (let [ten, sl] of Object.entries(data.counts)) {
          const row = document.createElement("tr");
          row.innerHTML = `<td>${ten}</td><td>${sl}</td>`;
          table.appendChild(row);
        }
      } else {
        table.innerHTML = `
          <tr class="empty-row">
            <td colspan="2">Không phát hiện được trái cây</td>
          </tr>
        `;
      }

      // Cập nhật thông tin
      const infoBox = document.getElementById("info-box");
      if (data.info && data.info.trim()) {
        infoBox.innerHTML = `
          <h4>Thông tin chi tiết</h4>
          <p>${data.info}</p>
        `;
        infoBox.classList.add("has-content");
      } else {
        clearInfo();
      }
    })
    .catch(error => {
      hideLoading();
      console.error('Error:', error);
      showError("Có lỗi xảy ra khi kết nối server. Vui lòng thử lại.");
    });
}



function resetUpload() {
  const fileInput = document.getElementById("file-input");
  const fileStatus = document.getElementById("file-status");
  const imageGrid = document.getElementById("image-grid");
  const imagePreviewContainer = document.getElementById("image-preview-container");
  const uploadContent = document.querySelector(".upload-content");
  
  // Reset file input
  fileInput.value = "";
  
  // Reset UI
  fileStatus.textContent = "Chưa có ảnh nào được chọn";
  imageGrid.innerHTML = "";
  imagePreviewContainer.classList.remove("show");
  uploadContent.style.display = "block";
  
  // Clear results
  clearResults();
}

function clearResults() {
  // Clear result images
  const resultArea = document.getElementById("result-image-area");
  resultArea.innerHTML = `
    <div class="no-result">
      <div class="no-result-icon">📷</div>
      <p>Chưa có ảnh kết quả</p>
      <small>Hãy tải ảnh và nhấn "Xử lý ảnh"</small>
    </div>
  `;
  resultArea.classList.remove("has-content");
  
  // Clear classification table
  const table = document.getElementById("classification-table");
  table.innerHTML = `
    <tr class="empty-row">
      <td colspan="2">Chưa có dữ liệu</td>
    </tr>
  `;
  
  // Clear info
  clearInfo();
}

// Thêm event listeners cho drag and drop
document.addEventListener('DOMContentLoaded', function() {
  const uploadArea = document.getElementById("upload-area");
  
  uploadArea.addEventListener('dragover', handleDragOver);
  uploadArea.addEventListener('dragleave', handleDragLeave);
  uploadArea.addEventListener('drop', handleDrop);
  
  // Thêm click event cho browse text
  const browseText = document.querySelector('.browse-text');
  if (browseText) {
    browseText.addEventListener('click', function() {
      document.getElementById('file-input').click();
    });
  }
});

// Thêm CSS animations
const style = document.createElement('style');
style.textContent = `
  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(-20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  @keyframes slideOut {
    from {
      opacity: 1;
      transform: translateY(0);
    }
    to {
      opacity: 0;
      transform: translateY(-20px);
    }
  }
`;
document.head.appendChild(style);
