// Simple module-level store for passing task ID to edit
let editingTaskId: number | null = null;

export const setEditingTaskId = (id: number | null) => {
  editingTaskId = id;
};

export const getEditingTaskId = (): number | null => {
  const id = editingTaskId;
  // Clear after reading so it doesn't persist
  editingTaskId = null;
  return id;
};
