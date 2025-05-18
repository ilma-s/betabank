declare module 'json2csv' {
  export interface ParserOptions<T> {
    fields?: Array<keyof T | {
      label: string;
      value: keyof T | ((row: T) => string);
      default?: string;
    }>;
    transforms?: Array<(item: T) => T>;
    defaultValue?: string;
    delimiter?: string;
    eol?: string;
    header?: boolean;
    includeEmptyRows?: boolean;
    withBOM?: boolean;
  }

  export class Parser<T> {
    constructor(opts?: ParserOptions<T>);
    parse(data: T[]): string;
  }
} 