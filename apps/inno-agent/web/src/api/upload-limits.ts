export const CSV_UPLOAD_LIMIT_BYTES = 50 * 1024 * 1024;
export const EXCEL_UPLOAD_LIMIT_BYTES = 25 * 1024 * 1024;
export const GENERAL_UPLOAD_LIMIT_BYTES = 50 * 1024 * 1024;
export const SKILL_UPLOAD_LIMIT_BYTES = 25 * 1024 * 1024;

export function uploadLimitForName(name: string): number {
	const extension = name.includes(".") ? `.${name.split(".").pop()!.toLowerCase()}` : "";
	if ([".csv", ".tsv", ".txt"].includes(extension)) return CSV_UPLOAD_LIMIT_BYTES;
	if ([".xlsx", ".xls"].includes(extension)) return EXCEL_UPLOAD_LIMIT_BYTES;
	if ([".zip", ".md"].includes(extension)) return SKILL_UPLOAD_LIMIT_BYTES;
	return GENERAL_UPLOAD_LIMIT_BYTES;
}

export function assertFileWithinUploadLimit(file: File): void {
	const maxBytes = uploadLimitForName(file.name);
	if (file.size > maxBytes) {
		throw new Error(`${file.name} 超过当前 ${Math.floor(maxBytes / (1024 * 1024))} MB 的安全上传上限`);
	}
}
