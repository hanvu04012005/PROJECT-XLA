function handleFiles(files) {
  const status = document.getElementById("file-status");
  status.textContent = files.length > 0 ? `${files.length} ảnh được chọn` : "Chưa có tệp nào được chọn";
}

function handleDrop(event) {
  event.preventDefault();
  const files = event.dataTransfer.files;
  document.getElementById("file-input").files = files;
  handleFiles(files);
}

function handleObjectChange() {
  const value = document.getElementById("object-select").value;
  const input = document.getElementById("object-name");
  const infoBox = document.getElementById("info-options");
  input.classList.toggle("hidden", value !== "doi-tuong");
  infoBox.classList.toggle("hidden", value !== "doi-tuong");
}

function handleFunctionChange() {
  const value = document.getElementById("function-select").value;
  const sub = document.getElementById("sub-function-select");
  sub.classList.toggle("hidden", value !== "sac-bien");
}

function handleProcess() {
  alert("Đang xử lý ảnh...");
}

function showInfo(type) {
  const infoBox = document.getElementById("info-box");
  infoBox.innerHTML = `<h4>Thông tin ${type === "cong-dung" ? "công dụng" : "dinh dưỡng"}...</h4>`;
}
