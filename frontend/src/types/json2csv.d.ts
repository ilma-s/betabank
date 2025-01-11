declare module 'json2csv' {
  export class Parser<T> {
    constructor(opts?: any);
    parse(data: T[]): string;
  }
} 