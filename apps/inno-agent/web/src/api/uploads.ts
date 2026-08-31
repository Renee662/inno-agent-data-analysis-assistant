import { ApiError } from "./client.js";
import { assertFileWithinUploadLimit } from "./upload-limits.js";

export interface RawUploadResult {
	fileName: string;
	mimeType: string;
	size: number;
	rawPath: string;
}

export async function uploadRawFile(file: File): Promise<RawUploadResult> {
	assertFileWithinUploadLimit(file);
	const params = new URLSearchParams({ fileName: file.name });
	const response = await fetch(`/api/l2/raw/upload?${params.toString()}`, {
		method: "POST",
		headers: { "Content-Type": file.type || "application/octet-stream" },
		body: file,
	});
	if (!response.ok) {
		const body = await response.json().catch(() => ({}));
		throw new ApiError(response.status, (body as Record<string, string>).error || response.statusText);
	}
	return response.json() as Promise<RawUploadResult>;
}
